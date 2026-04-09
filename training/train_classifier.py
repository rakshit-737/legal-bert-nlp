"""
Training pipeline for OptiBERT Legal Document Classifier
Implements the training loop from the paper (Section V):
  • AdamW with warmup scheduler
  • Gradient clipping
  • Best-model checkpointing by F1
  • Detailed per-epoch logging

Usage (CLI):
    python training/train_classifier.py

Usage (import):
    from training.train_classifier import TrainingPipeline
    pipeline = TrainingPipeline()
    pipeline.run()
"""

from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.classification_model import LegalDocumentClassifier, OptiBERTClassifier
from preprocessing.data_loader import get_splits

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LegalDocumentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TrainingPipeline:
    """End-to-end training pipeline for OptiBERT."""

    def __init__(
        self,
        model_name: str = None,
        output_dir: str = None,
        device: str = None,
    ):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.output_dir = Path(output_dir or config.CHECKPOINTS)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_accuracy": [], "val_f1": [],
            "val_precision": [], "val_recall": [],
        }

    # ── data ──────────────────────────────────────────────────────────────────

    def _make_loaders(
        self,
        splits: Dict,
        tokenizer,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        def _loader(split_key, shuffle):
            ds = LegalDocumentDataset(
                splits[split_key]["text"],
                splits[split_key]["label"],
                tokenizer,
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return (
            _loader("train",      shuffle=True),
            _loader("validation", shuffle=False),
            _loader("test",       shuffle=False),
        )

    # ── single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, scheduler, loss_fn) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(self.device)
            mask  = batch["attention_mask"].to(self.device)
            lbls  = batch["label"].to(self.device)

            logits = model(input_ids=ids, attention_mask=mask)
            if hasattr(logits, "logits"):
                logits = logits.logits

            loss = loss_fn(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAINING_CONFIG["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    # ── evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, model, loader, loss_fn) -> Dict:
        model.eval()
        total_loss, preds_all, labels_all = 0.0, [], []
        for batch in loader:
            ids   = batch["input_ids"].to(self.device)
            mask  = batch["attention_mask"].to(self.device)
            lbls  = batch["label"].to(self.device)

            logits = model(input_ids=ids, attention_mask=mask)
            if hasattr(logits, "logits"):
                logits = logits.logits

            total_loss += loss_fn(logits, lbls).item()
            preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())

        n = max(len(loader), 1)
        return {
            "loss":      total_loss / n,
            "accuracy":  accuracy_score(labels_all, preds_all),
            "f1":        f1_score(labels_all, preds_all, average="weighted", zero_division=0),
            "precision": precision_score(labels_all, preds_all, average="weighted", zero_division=0),
            "recall":    recall_score(labels_all, preds_all, average="weighted", zero_division=0),
        }

    # ── main ──────────────────────────────────────────────────────────────────

    def run(
        self,
        source: str = "auto",
        num_epochs: int = None,
        learning_rate: float = None,
        batch_size: int = None,
    ) -> Dict:
        """
        Load data → build model → train → evaluate → save artefacts.

        Returns the final test metrics dict.
        """
        num_epochs    = num_epochs    or config.TRAINING_CONFIG["num_train_epochs"]
        learning_rate = learning_rate or config.TRAINING_CONFIG["learning_rate"]
        batch_size    = batch_size    or config.TRAINING_CONFIG["per_device_train_batch_size"]

        print("\n" + "═" * 60)
        print("  OptiBERT Legal Document Classifier — Training")
        print("═" * 60)
        print(f"  Device   : {self.device}")
        print(f"  Model    : {self.model_name}")
        print(f"  Epochs   : {num_epochs}")
        print(f"  LR       : {learning_rate}")
        print(f"  Batch    : {batch_size}")
        print("═" * 60 + "\n")

        # 1 — data
        splits = get_splits(source=source)
        n_tr = len(splits["train"]["text"])
        n_va = len(splits["validation"]["text"])
        n_te = len(splits["test"]["text"])
        print(f"📊 Samples — train: {n_tr}  val: {n_va}  test: {n_te}\n")

        # 2 — tokeniser & loaders
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_loader, val_loader, test_loader = self._make_loaders(splits, tokenizer, batch_size)

        # 3 — model
        clf   = LegalDocumentClassifier(
            model_name=self.model_name,
            num_labels=len(config.CLASSIFICATION_LABELS),
            device=self.device,
            use_custom=True,
        )
        model = clf.get_model()

        # 4 — optimiser + scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.TRAINING_CONFIG["weight_decay"],
        )
        total_steps  = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * config.TRAINING_CONFIG["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss()

        # 5 — training loop
        best_f1   = 0.0
        best_path = self.output_dir / "best_optibert.pt"

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            print(f"┌── Epoch {epoch}/{num_epochs} ──────────────────────────")
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
            val_m      = self._evaluate(model, val_loader, loss_fn)
            elapsed    = time.time() - t0

            print(f"│  train_loss : {train_loss:.4f}")
            print(f"│  val_loss   : {val_m['loss']:.4f}")
            print(f"│  accuracy   : {val_m['accuracy']:.4f}")
            print(f"│  precision  : {val_m['precision']:.4f}")
            print(f"│  recall     : {val_m['recall']:.4f}")
            print(f"│  f1         : {val_m['f1']:.4f}")
            print(f"│  time       : {elapsed:.1f}s")

            # record
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_m["loss"])
            self.history["val_accuracy"].append(val_m["accuracy"])
            self.history["val_f1"].append(val_m["f1"])
            self.history["val_precision"].append(val_m["precision"])
            self.history["val_recall"].append(val_m["recall"])

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                torch.save(model.state_dict(), best_path)
                print(f"│  💾 New best model saved  (F1={best_f1:.4f})")
            print("└────────────────────────────────────────────────────")

        # 6 — test evaluation
        print("\n📐 Loading best checkpoint for test evaluation …")
        model.load_state_dict(torch.load(best_path, map_location=self.device))
        test_m = self._evaluate(model, test_loader, loss_fn)

        print("\n" + "═" * 60)
        print("  FINAL TEST RESULTS")
        print("═" * 60)
        for k, v in test_m.items():
            print(f"  {k:<12}: {v:.4f}")
        print("═" * 60)

        # 7 — save artefacts
        self._save_artefacts(model, tokenizer, test_m)
        return test_m

    # ── artefacts ─────────────────────────────────────────────────────────────

    def _save_artefacts(self, model, tokenizer, test_metrics: Dict):
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Training history JSON
        hist_path = results_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✅ Training history → {hist_path}")

        # Test metrics JSON
        metrics_path = results_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"✅ Test metrics     → {metrics_path}")

        # Save tokenizer for inference
        tok_path = Path(config.CHECKPOINTS) / "tokenizer"
        tokenizer.save_pretrained(str(tok_path))
        print(f"✅ Tokenizer        → {tok_path}")

        # Training curves PNG
        self._plot_curves(results_dir)

    def _plot_curves(self, results_dir: Path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.history["train_loss"]) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Accuracy
            ax = axes[0]
            ax.plot(epochs, self.history["val_accuracy"], "b-o", label="Validation Accuracy")
            ax.set_title("Training and Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1.05])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Loss
            ax = axes[1]
            ax.plot(epochs, self.history["train_loss"], "r-o", label="Training Loss")
            ax.plot(epochs, self.history["val_loss"],   "b-o", label="Validation Loss")
            ax.set_title("Training and Testing Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out = results_dir / "training_curves.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Training curves  → {out}")
        except ImportError:
            print("⚠️  matplotlib not available; skipping plot.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train OptiBERT on legal documents")
    parser.add_argument("--model",   default=None,  help="HuggingFace model name")
    parser.add_argument("--source",  default="auto", choices=["auto", "synthetic", "huggingface"])
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    args = parser.parse_args()

    pipeline = TrainingPipeline(model_name=args.model)
    pipeline.run(
        source=args.source,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
    )
