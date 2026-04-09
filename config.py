"""
Configuration file for Legal BERT Project
Enhanced based on:
  "Optimization of BERT Algorithms for Deep Contextual Analysis
   and Automation in Legal Document Processing" (ICCCNT 2024)
"""
import os

# ── Model Configuration ───────────────────────────────────────────────────────
MODEL_NAMES = {
    "legal":    "nlpaueb/legal-bert-base-uncased",  # domain-specific pre-training
    "generic":  "bert-base-uncased",
    "roberta":  "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

DEFAULT_MODEL = MODEL_NAMES["legal"]

# ── Optimized Hyper-parameters (paper Section IV-B) ──────────────────────────
TRAINING_CONFIG = {
    "learning_rate":                2e-5,
    "per_device_train_batch_size":  8,
    "per_device_eval_batch_size":   16,
    "num_train_epochs":             5,        # raised from 3 → 5 for convergence
    "warmup_ratio":                 0.1,      # warmup = 10 % of total steps
    "weight_decay":                 0.01,
    "logging_steps":                50,
    "eval_steps":                   200,
    "save_steps":                   200,
    "dropout":                      0.2,      # tuned dropout (paper Section IV-B)
    "max_grad_norm":                1.0,      # gradient clipping
    "attention_heads":              8,        # modified attention (paper Section IV-B)
    "seed":                         42,
}

# ── Classification Labels ─────────────────────────────────────────────────────
CLASSIFICATION_LABELS = {
    "contract": 0,
    "case":     1,
    "appeal":   2,
    "statute":  3,
}
LABEL_TO_CLASS = {v: k for k, v in CLASSIFICATION_LABELS.items()}

# ── NER Tags (BIO format) ─────────────────────────────────────────────────────
NER_TAGS = [
    "O",
    "B-PERSON",   "I-PERSON",
    "B-JUDGE",    "I-JUDGE",
    "B-DATE",     "I-DATE",
    "B-ORG",      "I-ORG",
    "B-CLAUSE",   "I-CLAUSE",
    "B-CASE",     "I-CASE",
    "B-STATUTE",  "I-STATUTE",
]
TAG_TO_ID = {tag: idx for idx, tag in enumerate(NER_TAGS)}
ID_TO_TAG  = {idx: tag for tag, idx in TAG_TO_ID.items()}

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS  = os.path.join(PROJECT_ROOT, "checkpoints")

for _d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, CHECKPOINTS]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
# Primary: lex_glue / scotus  (US Supreme Court opinion classification, 14 classes)
# Fallback: built-in legal corpus  (guaranteed to run offline)
DATASET_CONFIG = {
    "primary_name":   "lex_glue",
    "primary_task":   "scotus",          # multi-class legal classification
    "fallback":       "synthetic",       # always works
    "max_seq_length": 512,
    "train_size":     0.8,
    "val_size":       0.1,
    "test_size":      0.1,
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1"]

# ── Comparison baselines (from paper Table 2/3) ───────────────────────────────
BASELINE_RESULTS = {
    "DistilBERT": {"accuracy": 0.84, "f1": 0.83, "precision": 0.83, "recall": 0.84},
    "RoBERTa":    {"accuracy": 0.87, "f1": 0.86, "precision": 0.87, "recall": 0.86},
    "BERT-base":  {"accuracy": 0.89, "f1": 0.88, "precision": 0.88, "recall": 0.89},
    "OptiBERT":   {"accuracy": 0.92, "f1": 0.91, "precision": 0.91, "recall": 0.92},  # target
}
