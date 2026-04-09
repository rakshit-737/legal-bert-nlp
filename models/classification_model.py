"""
Text Classification Model — OptiBERT
Implements the optimized BERT classifier from the paper:
  "Optimization of BERT Algorithms for Deep Contextual Analysis
   and Automation in Legal Document Processing" (ICCCNT 2024)

Key architectural enhancements (Section IV-B):
  1. Modified multi-head attention (8 heads)
  2. Additional dropout (0.2) for regularisation
  3. Layer-norm–stabilised projection head
  4. Gradient-checkpointing support for memory efficiency
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Optimised BERT Classifier  (OptiBERT — paper architecture)
# ─────────────────────────────────────────────────────────────────────────────

class OptiBERTClassifier(nn.Module):
    """
    Custom BERT-based classifier with:
      • Fine-tuned attention mechanism (8 heads)
      • Dropout-regularised projection head
      • Layer-norm for stable training
      • Semantic pooling (CLS + mean pooling concat)
    """

    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL,
        num_labels: int = len(config.CLASSIFICATION_LABELS),
        dropout: float = config.TRAINING_CONFIG["dropout"],
        num_attention_heads: int = config.TRAINING_CONFIG["attention_heads"],
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size   # 768 for bert-base

        # Modified multi-head attention (paper Section IV-B)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)

        # Projection head
        self.dropout  = nn.Dropout(dropout)
        self.proj     = nn.Sequential(
            nn.Linear(hidden * 2, 512),   # CLS + mean pool → 2×hidden
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # BERT encoding
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_out = bert_out.last_hidden_state   # [B, L, H]

        # Modified attention (residual + norm)
        key_pad_mask = (attention_mask == 0)
        attn_out, _ = self.attention(seq_out, seq_out, seq_out, key_padding_mask=key_pad_mask)
        attn_out = self.attn_norm(attn_out + seq_out)  # residual

        # Dual pooling: CLS + mean
        cls_vec  = attn_out[:, 0, :]                                           # [B, H]
        mean_vec = (attn_out * attention_mask.unsqueeze(-1)).sum(1) \
                   / attention_mask.sum(1, keepdim=True).clamp(min=1e-9)       # [B, H]

        pooled = torch.cat([cls_vec, mean_vec], dim=-1)                        # [B, 2H]
        logits = self.proj(self.dropout(pooled))                               # [B, num_labels]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper (used by training pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class LegalDocumentClassifier:
    """
    High-level wrapper around OptiBERTClassifier.
    Also supports HuggingFace AutoModelForSequenceClassification as a baseline.
    """

    def __init__(
        self,
        model_name: str = None,
        num_labels: int = len(config.CLASSIFICATION_LABELS),
        device: str = None,
        use_custom: bool = True,
    ):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.num_labels = num_labels
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"⚙  Loading tokeniser: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if use_custom:
            print(f"⚙  Building OptiBERT classifier …")
            self.model = OptiBERTClassifier(
                model_name=self.model_name,
                num_labels=self.num_labels,
            )
        else:
            print(f"⚙  Loading AutoModelForSequenceClassification (baseline) …")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
            )

        self.model.to(self.device)

    # ── tokenisation ──────────────────────────────────────────────────────────

    def preprocess(self, texts: List[str], max_length: int = 512) -> Dict:
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = False,
    ) -> Tuple[List[int], List[float]]:
        self.model.eval()
        with torch.no_grad():
            inputs  = self.preprocess(texts)
            logits  = self._forward(inputs)
            preds   = torch.argmax(logits, dim=1).cpu().numpy()
            probs   = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
        return preds, probs

    def predict_single(self, text: str, return_proba: bool = False):
        preds, probs = self.predict([text], return_probabilities=True)
        label = config.LABEL_TO_CLASS.get(int(preds[0]), "unknown")
        return (label, float(probs[0])) if return_proba else label

    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Handle both OptiBERT (returns Tensor) and HF models (returns output object)."""
        out = self.model(**inputs)
        if hasattr(out, "logits"):
            return out.logits
        return out

    # ── accessors ─────────────────────────────────────────────────────────────

    def get_model(self)     -> nn.Module:      return self.model
    def get_tokenizer(self) -> AutoTokenizer:  return self.tokenizer

    def set_device(self, device: str):
        self.device = device
        self.model.to(device)
