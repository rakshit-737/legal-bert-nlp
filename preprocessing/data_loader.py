"""
Data loading utilities for legal document processing
"""
from datasets import load_dataset
import pandas as pd
from typing import Tuple, Dict
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_legal_dataset() -> Dict:
    """
    Load legal dataset from Hugging Face (lex_glue)
    Returns train, validation, test splits
    """
    print(f"Loading dataset: {config.DATASET_CONFIG['name']}")
    
    try:
        dataset = load_dataset(
            config.DATASET_CONFIG['name'],
            config.DATASET_CONFIG['task']
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Train samples: {len(dataset['train'])}")
        print(f"  - Validation samples: {len(dataset['validation'])}")
        print(f"  - Test samples: {len(dataset['test'])}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Falling back to creating dummy dataset...")
        return create_dummy_dataset()


def create_dummy_dataset() -> Dict:
    """
    Create dummy legal documents for testing/demo purposes
    """
    dummy_texts = [
        "This is a contract between Party A and Party B for services rendered.",
        "The court ruled in favor of the plaintiff on January 15, 2023.",
        "Section 101 of the statute provides guidelines for compliance.",
        "The defendant appealed the decision on grounds of procedural error.",
        "This document contains liability clauses and indemnification provisions.",
    ]
    
    dummy_labels = [0, 1, 3, 2, 0]  # 0: contract, 1: case, 2: appeal, 3: statute
    
    train_data = {"text": dummy_texts * 20, "label": dummy_labels * 20}
    val_data = {"text": dummy_texts * 5, "label": dummy_labels * 5}
    test_data = {"text": dummy_texts * 3, "label": dummy_labels * 3}
    
    from datasets import Dataset
    
    dataset = {
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data),
        "test": Dataset.from_dict(test_data),
    }
    
    return dataset


def prepare_classification_dataset(dataset):
    """Prepare dataset for text classification task"""
    def process_example(example):
        return {
            "text": example.get("text", ""),
            "label": example.get("label", 0),
        }
    
    processed = dataset.map(process_example)
    return processed


def prepare_ner_dataset(dataset):
    """Prepare dataset for Named Entity Recognition task"""
    
    def process_example(example):
        # Simple tokenization
        tokens = example.get("text", "").split()
        # Assign dummy tags (B-PERSON for first token, O for others)
        tags = []
        for i, token in enumerate(tokens):
            if i == 0:
                tags.append(config.TAG_TO_ID.get("B-PERSON", 0))
            else:
                tags.append(config.TAG_TO_ID.get("O", 0))
        
        return {
            "tokens": tokens,
            "ner_tags": tags,
        }
    
    processed = dataset.map(process_example)
    return processed


def prepare_similarity_dataset(dataset):
    """Prepare dataset for semantic similarity task"""
    # For similarity, we need pairs of documents
    texts = dataset["text"]
    pairs = []
    labels = []
    
    # Create similar pairs (same label) and dissimilar pairs (different labels)
    for i in range(0, len(texts)-1, 2):
        # Similar pair
        pairs.append({
            "text1": texts[i],
            "text2": texts[i+1] if i+1 < len(texts) else texts[0],
            "label": 1  # Similarity score (1=similar, 0=dissimilar)
        })
        
        # Dissimilar pair
        pairs.append({
            "text1": texts[i],
            "text2": texts[(i+2) % len(texts)],
            "label": 0
        })
    
    from datasets import Dataset
    return Dataset.from_dict({
        "text1": [p["text1"] for p in pairs],
        "text2": [p["text2"] for p in pairs],
        "label": [p["label"] for p in pairs]
    })


def get_dataset_splits(task: str = "classification"):
    """
    Load and prepare dataset for specific task
    
    Args:
        task: "classification", "ner", or "similarity"
    """
    dataset = load_legal_dataset()
    
    if task == "classification":
        return prepare_classification_dataset(dataset)
    elif task == "ner":
        return prepare_ner_dataset(dataset)
    elif task == "similarity":
        return prepare_similarity_dataset(dataset)
    else:
        raise ValueError(f"Unknown task: {task}")
