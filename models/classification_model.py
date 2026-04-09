"""
Text Classification Model using Legal BERT
Classifies legal documents into categories: contract, case, appeal, statute
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LegalDocumentClassifier:
    """
    Text classification for legal documents
    Supports: contract, case, appeal, statute
    """
    
    def __init__(self, model_name: str = None, num_labels: int = 4, device: str = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained model and tokenizer
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
    def preprocess(self, texts: List[str], max_length: int = 512) -> Dict:
        """Tokenize and prepare input for model"""
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encodings.items()}
    
    def predict(self, texts: List[str], return_probabilities: bool = False) -> Tuple[List[int], List[float]]:
        """
        Predict document class
        Returns: (predicted_labels, confidence_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.preprocess(texts)
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
        
        if return_probabilities:
            return predictions, probabilities
        return predictions, probabilities
    
    def predict_single(self, text: str, return_proba: bool = False):
        """Predict class for a single document"""
        pred, proba = self.predict([text], return_probabilities=True)
        
        if return_proba:
            return config.LABEL_TO_CLASS[pred[0]], proba[0]
        return config.LABEL_TO_CLASS[pred[0]]
    
    def get_model(self):
        """Return the PyTorch model for training"""
        return self.model
    
    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer
    
    def set_device(self, device: str):
        """Move model to device (cpu/cuda)"""
        self.device = device
        self.model.to(device)


class CustomBERTClassifier(nn.Module):
    """
    Custom classifier with attention and dropout for better performance
    """
    
    def __init__(self, model_name: str, num_labels: int = 4, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass with attention"""
        # Get BERT outputs
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply attention
        attention_output, _ = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=(attention_mask == 0)
        )
        
        # Use [CLS] token representation
        cls_output = attention_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(self.dropout(cls_output))
        return logits
