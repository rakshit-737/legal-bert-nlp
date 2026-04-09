"""
Training pipeline for Legal BERT models
Includes training loop, validation, and checkpointing
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.classification_model import LegalDocumentClassifier


class LegalDocumentDataset(Dataset):
    """Custom dataset for legal documents"""
    
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class TrainingPipeline:
    """
    Complete training pipeline for classification
    """
    
    def __init__(self, model_name: str = None, output_dir: str = "./results", device: str = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        self.training_history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_f1": []
        }
    
    def prepare_datasets(self, train_texts: list, train_labels: list,
                        val_texts: list, val_labels: list,
                        batch_size: int = config.TRAINING_CONFIG["per_device_train_batch_size"]):
        """Create DataLoaders for train and validation"""
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        train_dataset = LegalDocumentDataset(train_texts, train_labels, tokenizer)
        val_dataset = LegalDocumentDataset(val_texts, val_labels, tokenizer)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False
        )
        
        return train_loader, val_loader, tokenizer
    
    def train_epoch(self, model, train_loader, optimizer, loss_fn) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item():.4f})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, model, val_loader, loss_fn) -> Dict:
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        return {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def train(self, train_texts: list, train_labels: list,
              val_texts: list, val_labels: list,
              num_epochs: int = config.TRAINING_CONFIG["num_train_epochs"],
              learning_rate: float = config.TRAINING_CONFIG["learning_rate"],
              batch_size: int = config.TRAINING_CONFIG["per_device_train_batch_size"]):
        """
        Full training pipeline
        """
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
        
        # Prepare dataloaders
        train_loader, val_loader, tokenizer = self.prepare_datasets(
            train_texts, train_labels, val_texts, val_labels, batch_size
        )
        
        # Initialize model
        self.classifier = LegalDocumentClassifier(
            model_name=self.model_name,
            num_labels=len(config.CLASSIFICATION_LABELS),
            device=self.device
        )
        model = self.classifier.get_model()
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        best_f1 = 0
        best_model_path = self.output_dir / "best_model.pt"
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn)
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Evaluate
            eval_metrics = self.evaluate(model, val_loader, loss_fn)
            print(f"  Val Loss: {eval_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Val F1: {eval_metrics['f1']:.4f}")
            print(f"  Val Precision: {eval_metrics['precision']:.4f}")
            print(f"  Val Recall: {eval_metrics['recall']:.4f}")
            
            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["eval_loss"].append(eval_metrics['loss'])
            self.training_history["eval_accuracy"].append(eval_metrics['accuracy'])
            self.training_history["eval_f1"].append(eval_metrics['f1'])
            
            # Save best model
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                torch.save(model.state_dict(), best_model_path)
                print(f"  💾 Best model saved (F1: {best_f1:.4f})")
        
        print(f"\n✅ Training complete!")
        print(f"📁 Model saved to: {best_model_path}")
        
        return self.classifier, self.training_history
    
    def save_model(self, output_path: str):
        """Save trained model"""
        if self.classifier:
            self.classifier.get_model().save_pretrained(output_path)
            self.classifier.get_tokenizer().save_pretrained(output_path)
            print(f"✅ Model saved to {output_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        self.classifier = LegalDocumentClassifier(
            model_name=model_path,
            device=self.device
        )
        return self.classifier


def train_with_transformers(train_dataset, val_dataset, model_name: str = None,
                            output_dir: str = "./results"):
    """
    Alternative: Use HuggingFace Trainer API for simpler training
    """
    from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
    
    model_name = model_name or config.DEFAULT_MODEL
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(config.CLASSIFICATION_LABELS)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=config.TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=config.TRAINING_CONFIG["per_device_eval_batch_size"],
        num_train_epochs=config.TRAINING_CONFIG["num_train_epochs"],
        weight_decay=config.TRAINING_CONFIG["weight_decay"],
        logging_steps=config.TRAINING_CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=config.TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=config.TRAINING_CONFIG["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=AutoTokenizer.from_pretrained(model_name),
    )
    
    # Train
    trainer.train()
    
    return trainer
