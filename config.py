"""
Configuration file for Legal BERT Project
"""
import os

# Model Configuration
MODEL_NAMES = {
    "legal": "nlpaueb/legal-bert-base-uncased",
    "generic": "bert-base-uncased",
    "roberta": "roberta-base"
}

DEFAULT_MODEL = MODEL_NAMES["legal"]

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "eval_steps": 500,
    "save_steps": 500,
}

# Classification Tasks
CLASSIFICATION_LABELS = {
    "contract": 0,
    "case": 1,
    "appeal": 2,
    "statute": 3,
}

LABEL_TO_CLASS = {v: k for k, v in CLASSIFICATION_LABELS.items()}

# NER Tags (BIO format)
NER_TAGS = {
    "O": 0,  # Outside
    "B-PERSON": 1,  # Beginning of Person
    "I-PERSON": 2,  # Inside Person
    "B-JUDGE": 3,
    "I-JUDGE": 4,
    "B-DATE": 5,
    "I-DATE": 6,
    "B-ORGANIZATION": 7,
    "I-ORGANIZATION": 8,
    "B-CLAUSE": 9,
    "I-CLAUSE": 10,
}

TAG_TO_ID = {tag: idx for idx, tag in enumerate(NER_TAGS.keys())}
ID_TO_TAG = {idx: tag for tag, idx in TAG_TO_ID.items()}

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset Configuration
DATASET_CONFIG = {
    "name": "lex_glue",
    "task": "ecthr_a",  # European Court of Human Rights case outcome prediction
}

# Evaluation Metrics
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1"]
