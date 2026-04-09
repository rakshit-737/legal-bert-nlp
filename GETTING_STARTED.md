# 🚀 GETTING STARTED - Legal BERT NLP

## 📋 Table of Contents
1. [Installation](#installation)
2. [First Run](#first-run)
3. [Understanding the Basics](#understanding-the-basics)
4. [Common Tasks](#common-tasks)
5. [Troubleshooting](#troubleshooting)

---

## 💻 Installation

### Step 1: Verify Python Version
```bash
python --version  # Should be 3.9+
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **torch** - Deep learning framework
- **transformers** - BERT models from HuggingFace
- **datasets** - Dataset loading
- **scikit-learn** - Metrics & evaluation
- **streamlit** - Interactive UI
- **sentence-transformers** - Semantic similarity
- **pdfplumber** - PDF text extraction (optional)

### Step 4: Verify Installation
```bash
python quickstart.py
```

This will:
- ✅ Check all dependencies
- ✅ Test GPU availability
- ✅ Load sample models
- ✅ Show quick examples

---

## 🎯 First Run

### Option 1: Interactive Web UI (Easiest)
```bash
streamlit run app/streamlit_app.py
```
Then open http://localhost:8501

**What you get:**
- 📄 Upload & classify documents
- 🔍 Extract entities
- 🔗 Compare documents
- 📊 Batch processing
- 💾 Export results

### Option 2: Python Script
```bash
python examples.py
```

This runs 6 complete examples:
1. Basic inference
2. Full pipeline
3. Model training
4. Evaluation
5. Similarity analysis
6. NER demo

### Option 3: Interactive Quickstart
```bash
python quickstart.py
```

Choose from menu:
- View quick examples
- Run Streamlit UI
- Train custom models
- Read documentation

---

## 📚 Understanding the Basics

### The Three Main Models

#### 1. **Classification** 🏷️
*Categorize document type*

```python
from models.classification_model import LegalDocumentClassifier

classifier = LegalDocumentClassifier()

# Types: contract, case, appeal, statute
label, confidence = classifier.predict_single(your_text, return_proba=True)
print(f"Type: {label} ({confidence:.1%})")
```

**When to use:**
- Organize large document repositories
- Route documents to appropriate department
- Detect document type automatically

---

#### 2. **Named Entity Recognition (NER)** 🔍
*Extract important entities*

```python
from models.ner_model import LegalEntityRecognizer

ner = LegalEntityRecognizer()

# Entities: PERSON, JUDGE, DATE, ORGANIZATION, CLAUSE
entities = ner.extract_entities_by_type(your_text)
```

**Entities found:**
- 👤 **PERSON** - Individual names
- ⚖️ **JUDGE** - Judge/magistrate names  
- 📅 **DATE** - Important dates
- 🏢 **ORGANIZATION** - Company/institution names
- 📋 **CLAUSE** - Contract clauses

**When to use:**
- Extract key parties from contracts
- Find all dates in documents
- Summary generation
- Key information highlighting

---

#### 3. **Semantic Similarity** 🔗
*Compare document relatedness*

```python
from models.similarity_model import LegalSemanticSimilarity

similarity = LegalSemanticSimilarity()

# Compare two documents
score = similarity.similarity(doc1, doc2)  # 0 (different) to 1 (identical)

# Find similar documents
results = similarity.most_similar(query, corpus, top_k=5)
```

**When to use:**
- Find duplicate clauses
- Locate similar contracts
- Document clustering
- Deduplication

---

### The Complete Pipeline

```python
from inference.processor import LegalDocumentProcessor

processor = LegalDocumentProcessor()

# One-stop shop for all tasks!
result = processor.process_document(your_text)

# Contains:
# - classification: {label, confidence}
# - entities: {PERSON, JUDGE, DATE, ...}
# - cleaned_text: Processed version
```

---

## 🛠️ Common Tasks

### Task 1: Classify a Single Document

```python
from models.classification_model import LegalDocumentClassifier

classifier = LegalDocumentClassifier()

text = "This agreement between ABC Corp and XYZ Inc..."
label, confidence = classifier.predict_single(text, return_proba=True)

print(f"Document: CONTRACT")
print(f"Confidence: 95.2%")
```

### Task 2: Extract Key Information

```python
from inference.processor import LegalDocumentProcessor, DocumentSummarizer

processor = LegalDocumentProcessor()
summarizer = DocumentSummarizer(processor)

summary = summarizer.get_document_summary(text)

print(f"Type: {summary['type']}")
print(f"Key entities: {summary['key_entities']}")
print(f"Key clauses: {summary['key_clauses']}")
print(f"Reading time: {summary['estimated_reading_time_minutes']} min")
```

### Task 3: Compare Two Contracts

```python
from inference.processor import LegalDocumentProcessor

processor = LegalDocumentProcessor()

similarity = processor.calculate_similarity(contract1, contract2)

print(f"Similarity: {similarity:.3f}")
print(f"Match: {similarity*100:.1f}%")

if similarity > 0.8:
    print("⚠️  These contracts are very similar!")
```

### Task 4: Batch Process Files

```python
from inference.processor import batch_process_documents, LegalDocumentProcessor

processor = LegalDocumentProcessor()
documents = ["Doc 1", "Doc 2", "Doc 3", ...]

# Classify all
results = batch_process_documents(
    documents, 
    processor, 
    task="classify"
)

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
```

### Task 5: Custom Training

```python
from training.train_classifier import TrainingPipeline

pipeline = TrainingPipeline()

# Your labeled data
train_texts = ["Contract between...", "Court decision...", ...]
train_labels = [0, 1, ...]  # 0=contract, 1=case, 2=appeal, 3=statute
val_texts = [...]
val_labels = [...]

# Train
classifier, history = pipeline.train(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    num_epochs=3
)
```

---

## ⚡ Performance Tips

### Tip 1: Use GPU (if available)
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())  # Should be True

# Use GPU
classifier = LegalDocumentClassifier(device="cuda")
# ~10x speedup!
```

### Tip 2: Batch Processing
```python
# Instead of:
for doc in documents:
    classifier.predict_single(doc)  # Slow!

# Do this:
texts = [d for d in documents]
predictions = classifier.predict(texts)  # Fast!
```

### Tip 3: Cache Embeddings
```python
from models.similarity_model import LegalSemanticSimilarity

similarity = LegalSemanticSimilarity()

# First call: slow (computes embedding)
score1 = similarity.similarity(doc1, doc2)

# Second call: faster (cached)
score2 = similarity.similarity(doc1, doc2)

# Clear cache if needed
similarity.clear_cache()
```

### Tip 4: Chunk Long Documents
```python
from inference.processor import LegalDocumentProcessor

processor = LegalDocumentProcessor()

# Split document into chunks
chunks = processor.chunk_document(very_long_text, chunk_size=512)

# Process each chunk
for chunk in chunks:
    result = processor.classify_document(chunk)
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'transformers'"

**Solution:**
```bash
pip install transformers
# Or reinstall everything:
pip install -r requirements.txt
```

---

### Problem: "CUDA out of memory"

**Solution:**
```python
# Use smaller batch size
batch_size = 4  # Instead of 8

# Or use CPU
classifier = LegalDocumentClassifier(device="cpu")
```

---

### Problem: "Model download takes forever"

**Solution:**
```bash
# Models are cached - first run is slowest
# Subsequent runs are instant

# Or pre-download models:
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
"
```

---

### Problem: "Streamlit port 8501 already in use"

**Solution:**
```bash
# Use different port
streamlit run app/streamlit_app.py --server.port 8502
```

---

### Problem: "Out of disk space"

**Models are ~500MB each**

```bash
# Check disk space
df -h  # macOS/Linux
dir   # Windows

# Save space: use smaller model
# Instead of: legal-bert (340MB)
# Use: DistilBERT (268MB)
```

---

## 📊 What Model to Use?

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| **legal-bert** | Medium | Highest ⭐⭐⭐⭐⭐ | 340MB | Production \| Legal tasks |
| **bert-base** | Medium | High ⭐⭐⭐⭐ | 440MB | General tasks |
| **distilbert** | Fast ⚡ | Good ⭐⭐⭐ | 268MB | Mobile \| Speed critical |
| **roberta-base** | Medium | High ⭐⭐⭐⭐ | 498MB | Alternative |

**Recommendation:** Start with `legal-bert` (default) - best for legal documents!

---

## 📖 Next Steps

1. **Try the UI**: `streamlit run app/streamlit_app.py`
2. **Run examples**: `python examples.py`
3. **Read full docs**: `README.md`
4. **Customize**: Edit configs in `config.py`
5. **Train**: Use your own labeled data!

---

## 💡 Quick Reference Card

```python
# Classification
classifier.predict_single(text, return_proba=True)  # → (label, confidence)
classifier.predict(texts)  # → (predictions, confidences)

# NER
ner.predict(text)  # → [(token, type, conf), ...]
ner.extract_entities_by_type(text)  # → {type: [tokens], ...}

# Similarity
similarity.similarity(doc1, doc2)  # → 0.0-1.0
similarity.most_similar(query, corpus, top_k=5)  # → [(doc, score), ...]

# Pipeline
processor.process_document(text)  # → {classification, entities, ...}
processor.classify_document(text)  # → {label, confidence}
processor.extract_entities(text)  # → entities

# Evaluation
evaluator.get_metrics()  # → {accuracy, f1, precision, recall}
evaluator.plot_confusion_matrix()  # → visualization
```

---

## 🎓 Learning Resources

- 📖 **Paper**: [Legal-BERT](https://arxiv.org/abs/2010.02559)
- 🤗 **HuggingFace**: https://huggingface.co/nlpaueb
- 🔗 **Transformers**: https://huggingface.co/docs/transformers/
- 🐍 **PyTorch**: https://pytorch.org/tutorials/

---

📞 **Need help?**
- Check examples.py for complete working code
- Review README.md for detailed docs
- Check config.py for configuration options

**Happy coding! ⚖️📜**
