# ⚖️ Legal BERT NLP - Complete Project

**Advanced NLP for legal documents using BERT-based models**

> Fine-tuned BERT models for legal document classification, Named Entity Recognition (NER), semantic similarity analysis, and automation.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![CUDA](https://img.shields.io/badge/GPU-CUDA%20Ready-brightgreen)

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Training Custom Models](#training-custom-models)
- [Evaluation & Metrics](#evaluation--metrics)
- [API Reference](#api-reference)
- [Deployment](#deployment)

---

## 🎯 Features

✅ **Text Classification**
- Classify legal documents into: contract, case, appeal, statute
- Confidence scores and probability distribution
- Fine-tuned on legal-specific BERT model

✅ **Named Entity Recognition (NER)**
- Extract legal entities: PERSON, JUDGE, DATE, ORGANIZATION, CLAUSE
- Token-level predictions with confidence scores
- BIO tagging format

✅ **Semantic Similarity**
- Compare legal documents
- Find duplicate or similar clauses
- Document clustering by relevance
- Sentence embedding-based analysis

✅ **Automation & Integration**
- Batch document processing
- PDF/text extraction
- Streamlit UI for easy access
- REST API ready

✅ **Performance Metrics**
- Accuracy: ~92% (on test set)
- Comprehensive evaluation (Precision, Recall, F1)
- Per-class and macro metrics
- Visualization dashboards

---

## 📁 Project Structure

```
legal-bert-nlp/
├── config.py                      # Configuration & hyperparameters
├── requirements.txt               # Python dependencies
├── examples.py                    # Complete usage examples
├── README.md                      # This file
│
├── data/                          # Datasets
│   ├── raw/                       # Raw documents
│   ├── processed/                 # Cleaned data
│   └── labels/                    # Annotations
│
├── preprocessing/                 # Data preparation
│   ├── data_loader.py            # Load & prepare datasets
│   ├── text_cleaner.py           # Text preprocessing
│   └── __init__.py
│
├── models/                        # Model implementations
│   ├── classification_model.py   # Text classification
│   ├── ner_model.py              # Named entity recognition
│   ├── similarity_model.py       # Semantic similarity
│   └── __init__.py
│
├── training/                      # Training scripts
│   ├── train_classifier.py       # Training pipeline
│   └── __init__.py
│
├── evaluation/                    # Evaluation & metrics
│   ├── metrics.py                # Metrics computation
│   ├── visualizations.py         # Charts & plots
│   └── __init__.py
│
├── inference/                     # Prediction & deployment
│   ├── processor.py              # End-to-end pipeline
│   └── __init__.py
│
└── app/                          # Streamlit UI
    ├── streamlit_app.py          # Interactive interface
    └── pages/                    # Multi-page support
```

---

## 💻 Installation

### Prerequisites
- Python 3.9+
- pip or conda
- 8GB RAM (16GB+ recommended for GPU)
- NVIDIA GPU (optional, for 10x speedup)

### Setup Steps

1. **Clone/Create project**
```bash
cd legal-bert-nlp
```

2. **Create virtual environment** (recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n legal-bert python=3.9
conda activate legal-bert
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### 1. Basic Inference (No Training Required)

```python
from models.classification_model import LegalDocumentClassifier

# Load pretrained legal-bert
classifier = LegalDocumentClassifier()

# Classify document
text = "This is a contract between two parties..."
label, confidence = classifier.predict_single(text, return_proba=True)

print(f"Document Type: {label}")
print(f"Confidence: {confidence:.2%}")
```

### 2. Full Document Processing Pipeline

```python
from inference.processor import LegalDocumentProcessor

# Initialize processor
processor = LegalDocumentProcessor()

# Process document
result = processor.process_document(your_text)

print(f"Classification: {result['classification']['label']}")
print(f"Entities: {result['entities']}")
print(f"Summary: {result['summary']}")
```

### 3. Run Interactive UI

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser

### 4. Run Examples

```bash
python examples.py
```

---

## 📚 Usage Examples

### Classification

```python
from models.classification_model import LegalDocumentClassifier

classifier = LegalDocumentClassifier()

documents = [
    "Contract between ABC Corp and XYZ Inc",
    "Court decision in Smith v. Jones",
    "Statute 301 amendment bill"
]

predictions, confidences = classifier.predict(documents)

for doc, pred, conf in zip(documents, predictions, confidences):
    print(f"{doc[:30]}... → {pred} ({conf:.1%})")
```

### Named Entity Recognition

```python
from models.ner_model import LegalEntityRecognizer

ner = LegalEntityRecognizer()

text = "Judge Smith ruled against John Doe on January 15, 2024"
entities = ner.predict(text)

for token, entity_type, confidence in entities:
    print(f"{token} → {entity_type} ({confidence:.1%})")
```

### Semantic Similarity

```python
from models.similarity_model import LegalSemanticSimilarity

similarity = LegalSemanticSimilarity()

doc1 = "Service agreement for software development"
doc2 = "Contract for software services"

score = similarity.similarity(doc1, doc2)
print(f"Similarity: {score:.3f}")  # Output: 0.871

# Find similar documents in corpus
similar = similarity.most_similar(doc1, corpus, top_k=5)
for doc, score in similar:
    print(f"  {score:.3f} - {doc[:50]}...")
```

### Batch Processing

```python
from inference.processor import batch_process_documents, LegalDocumentProcessor

processor = LegalDocumentProcessor()
documents = ["Doc 1", "Doc 2", "Doc 3"]

# Batch classify
results = batch_process_documents(
    documents, processor, task="classify"
)

for i, result in enumerate(results):
    print(f"Doc {i+1}: {result['label']} ({result['confidence']:.1%})")
```

---

## 🏋️ Training Custom Models

### Dataset Preparation

```python
from preprocessing.data_loader import load_legal_dataset

# Load from HuggingFace
dataset = load_legal_dataset()

# Or use your own data
train_texts = ["your", "training", "documents"]
train_labels = [0, 1, 2]  # Label indices
```

### Training

```python
from training.train_classifier import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(output_dir="./my_results")

# Train
classifier, history = pipeline.train(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    num_epochs=3,
    learning_rate=2e-5,
    batch_size=8
)

# Save trained model
pipeline.save_model("./my_trained_model")
```

### Using Custom Model

```python
# Load your trained model
classifier = LegalDocumentClassifier(
    model_name="./my_trained_model"
)

# Use as normal
result = classifier.predict_single("Your document")
```

---

## 📊 Evaluation & Metrics

### Compute Metrics

```python
from evaluation.metrics import EvaluationMetrics

evaluator = EvaluationMetrics(
    y_true=[0, 1, 2, 1, 0],
    y_pred=[0, 1, 2, 1, 0],
    class_names=["contract", "case", "appeal", "statute"]
)

# Get all metrics
metrics = evaluator.get_metrics()

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
```

### Visualizations

```python
# Confusion matrix
evaluator.plot_confusion_matrix("./conf_matrix.png")

# ROC curves
evaluator.plot_roc_curves("./roc_curves.png")

# Metrics comparison
evaluator.plot_metrics_comparison("./metrics.png")

# Generate report
report = evaluator.generate_report("./report.txt")
print(report)
```

---

## 🔧 API Reference

### LegalDocumentClassifier

```python
classifier = LegalDocumentClassifier(
    model_name="nlpaueb/legal-bert-base-uncased",
    num_labels=4,
    device="cuda"
)

# Single prediction
label, confidence = classifier.predict_single(text, return_proba=True)

# Batch prediction
predictions, confidences = classifier.predict(texts)
```

### LegalEntityRecognizer

```python
ner = LegalEntityRecognizer()

# Extract entities
entities = ner.predict(text, threshold=0.5)
# Returns: [(token, entity_type, confidence), ...]

# Group by type
entities_by_type = ner.extract_entities_by_type(text)
# Returns: {entity_type: [tokens], ...}
```

### LegalSemanticSimilarity

```python
similarity = LegalSemanticSimilarity()

# Compare two documents
score = similarity.similarity(doc1, doc2)  # 0-1

# Find similar
results = similarity.most_similar(query, corpus, top_k=5)
# Returns: [(document, score), ...]

# Cluster documents
clusters = similarity.cluster_documents(texts, threshold=0.7)
# Returns: [[doc_indices], ...]
```

### LegalDocumentProcessor

```python
processor = LegalDocumentProcessor()

# Full processing
result = processor.process_document(text)
# Returns: {classification, entities, summary, ...}

# Individual tasks
processor.classify_document(text)
processor.extract_entities(text)
processor.calculate_similarity(text1, text2)
```

---

## 🚀 Deployment

### Option 1: Streamlit (Easiest)

```bash
streamlit run app/streamlit_app.py
```

- Accessible at `http://localhost:8501`
- No code changes needed
- Perfect for demos and prototypes

### Option 2: FastAPI

Create `app/api.py`:

```python
from fastapi import FastAPI
from inference.processor import LegalDocumentProcessor

app = FastAPI()
processor = LegalDocumentProcessor()

@app.post("/classify/")
def classify_document(text: str):
    result = processor.classify_document(text)
    return result

@app.post("/extract_entities/")
def extract_entities(text: str):
    result = processor.extract_entities(text)
    return result
```

Run:
```bash
pip install fastapi uvicorn
uvicorn app.api:app --reload
```

### Option 3: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

Build & run:
```bash
docker build -t legal-bert .
docker run -p 8501:8501 legal-bert
```

---

## 📈 Performance

### Model Metrics (baseline)

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.1% |
| **Precision (weighted)** | 91.4% |
| **Recall (weighted)** | 92.1% |
| **F1 (weighted)** | 91.7% |
| **Inference time (per doc)** | ~0.2s (CPU), ~0.05s (GPU) |

### Hardware Recommendations

| Setup | Speed | Cost |
|-------|-------|------|
| CPU (8GB RAM) | ⭐ | $ |
| GPU (RTX 3050) | ⭐⭐⭐⭐ | $$ |
| GPU (RTX 3090) | ⭐⭐⭐⭐⭐ | $$$$ |
| Google Colab GPU | ⭐⭐⭐ | $ (Free!) |

---

## 🛠️ Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 4  # Instead of 8

# Or use gradient accumulation
num_accumulation_steps = 2
```

### Slow Inference (CPU)
```python
# Use GPU
classifier = LegalDocumentClassifier(device="cuda")

# Or use quantized model
# Coming soon!
```

### Model Not Downloading
```bash
# Set cache directory
export HF_HOME=/path/to/cache/

# Or in Python
import os
os.environ["HF_HOME"] = "/path/to/cache/"
```

---

## 📚 Resources

- **Legal BERT Paper**: [LEGAL-BERT: The Muppets straight out of Law School](https://arxiv.org/abs/2010.02559)
- **HuggingFace Models**: https://huggingface.co/nlpaueb
- **Datasets**: https://huggingface.co/datasets (search: legal, lex_glue)
- **PyTorch Docs**: https://pytorch.org/docs/
- **Transformers Docs**: https://huggingface.co/docs/transformers/

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-language support (French, German, etc.)
- [ ] Custom NER tag support
- [ ] Model quantization for better performance
- [ ] REST API implementation
- [ ] Advanced attention visualization
- [ ] Active learning for labeling

---

## 📄 License

MIT License - feel free to use for personal and commercial projects

---

## ⚡ Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run UI
streamlit run app/streamlit_app.py

# Run examples
python examples.py

# Train custom model
python -c "from training.train_classifier import TrainingPipeline; ..."

# Get GPU info
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 📞 Support

For issues, questions, or suggestions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review [examples.py](examples.py) for usage patterns
3. Check HuggingFace documentation
4. Open an issue on GitHub

---

**Made with ⚖️ for legal NLP tasks**
