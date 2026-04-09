"""
Named Entity Recognition (NER) Model
Identifies legal entities: PERSON, JUDGE, DATE, ORGANIZATION, CLAUSE
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LegalEntityRecognizer:
    """
    Token-level classification for Named Entity Recognition
    Identifies legal entities in documents
    """
    
    def __init__(self, model_name: str = None, num_labels: int = None, device: str = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.num_labels = num_labels or len(config.TAG_TO_ID)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tag mappings
        self.tag_to_id = config.TAG_TO_ID
        self.id_to_tag = config.ID_TO_TAG
        
        # Load model
        print(f"Loading NER model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
    
    def tokenize_and_align_labels(self, tokens: List[str], tags: List[int], max_length: int = 512):
        """
        Tokenize text with subword tokens and align NER labels
        """
        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = []
        word_ids = tokenized_inputs.word_ids()
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                labels.append(tags[word_idx])
            else:
                # Subword token continuation
                labels.append(tags[word_idx])
            previous_word_idx = word_idx
        
        tokenized_inputs["labels"] = torch.tensor([labels])
        return {k: v.to(self.device) for k, v in tokenized_inputs.items()}
    
    def predict(self, text: str, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Predict named entities in text
        Returns: [(token, entity_type, confidence), ...]
        """
        # Naive tokenization for demo
        tokens = text.split()
        
        # Convert to inputs
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        word_ids = encoded.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0]
            probabilities = torch.softmax(logits, dim=2)[0]
        
        # Extract entities
        entities = []
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(tokens):
                continue
            
            tag_id = predictions[idx].item()
            confidence = probabilities[idx, tag_id].item()
            
            if confidence >= threshold and tag_id != self.tag_to_id.get("O", 0):
                tag_name = self.id_to_tag.get(tag_id, "O")
                entities.append((tokens[word_idx], tag_name, confidence))
        
        return entities
    
    def extract_entities_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities grouped by type
        Returns: {entity_type: [list of entities]}
        """
        entities = self.predict(text)
        result = {}
        
        for token, tag, _ in entities:
            entity_type = tag.split("-")[1] if "-" in tag else tag
            if entity_type not in result:
                result[entity_type] = []
            result[entity_type].append(token)
        
        return result
    
    def get_model(self):
        """Return the model for training"""
        return self.model
    
    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer


class AttentionNERModel(nn.Module):
    """
    Enhanced NER model with attention mechanism
    """
    
    def __init__(self, model_name: str, num_labels: int = None, dropout: float = 0.1):
        super().__init__()
        num_labels = num_labels or len(config.TAG_TO_ID)
        self.bert = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=12,
            dropout=dropout,
            batch_first=True
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass with attention"""
        # Get BERT embeddings
        bert_output = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        
        # Apply attention
        attention_output, _ = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=(attention_mask == 0)
        )
        
        # Classify
        logits = self.classifier(self.dropout(attention_output))
        return logits
