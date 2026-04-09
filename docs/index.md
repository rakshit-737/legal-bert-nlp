# 🏛️ Legal BERT NLP - Complete Documentation

**Status:** ✅ 100% Complete | Production Ready | Open Source

## 🚀 Live Applications

### Try the App
- **🌐 Streamlit Cloud**: [legal-bert-nlp.streamlit.app](https://legal-bert-nlp.streamlit.app)
  - Full interactive interface
  - Real-time document analysis
  - No installation required

### Features
- 📄 **Document Classification** - Identify document type (contract, case, appeal, statute)
- 🔍 **Named Entity Recognition** - Extract legal entities and key terms
- 🔗 **Similarity Analysis** - Compare documents, find duplicates
- 📊 **Document Summarization** - Extract key information automatically
- ⚡ **Batch Processing** - Process multiple documents at once

## 📚 Documentation

### Getting Started
1. [Quick Start Guide](GETTING_STARTED.md) - Set up and run locally in 5 minutes
2. [Installation](README.md#installation) - Detailed setup instructions
3. [API Reference](API.md) - Complete function documentation

### Usage Examples
- [Basic Example](examples.py) - Simple classification
- [NER Example](examples.py) - Entity extraction
- [Batch Processing](examples.py) - Process multiple documents
- [Model Training](examples.py) - Fine-tune on custom data

### Deployment
- [Streamlit Cloud Setup](DEPLOYMENT_GUIDE.md#quick-start-streamlit-cloud)
- [Docker Deployment](DEPLOYMENT_GUIDE.md#option-2-docker--cloud-run)
- [AWS EC2 Setup](DEPLOYMENT_GUIDE.md#option-3-aws-ec2)

## 🔧 Technical Details

### Architecture
- **Base Model**: Legal-BERT (fine-tuned for law)
- **Framework**: PyTorch + Transformers
- **Interface**: Streamlit
- **Languages**: Python 3.9+

### Models Included
1. **Classification Model** - 4-class document type classifier
2. **NER Model** - Named entity recognizer (11 entity types)
3. **Similarity Model** - Semantic similarity calculator
4. **Advanced Optimizations** - Ensemble methods, mixed precision training

### Performance
- ✅ GPU-accelerated (CUDA compatible)
- ✅ Batch processing support
- ✅ Embedding caching for efficiency
- ✅ Model caching for rapid inference

## 📊 Research

This implementation extends research on "Optimization of BERT Algorithms for Deep Contextual Analysis and Automation in Legal Document Processing" with:

- Custom attention mechanisms for legal content
- Domain-specific masking strategies
- Contrastive learning for document similarity
- Production-grade deployment infrastructure

## 🛠️ Installation

### Option 1: Streamlit Cloud (No Setup Required)
Just visit: **[legal-bert-nlp.streamlit.app](https://legal-bert-nlp.streamlit.app)**

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/legal-bert-nlp.git
cd legal-bert-nlp

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/streamlit_app.py
```

### Option 3: Docker
```bash
docker build -t legal-bert-nlp .
docker run -p 8501:8501 legal-bert-nlp
# Visit http://localhost:8501
```

## 📝 Usage Examples

### Classification
```python
from inference.processor import LegalDocumentProcessor

processor = LegalDocumentProcessor()
result = processor.classify_document("This is a service contract...")
print(f"Type: {result['label']}, Confidence: {result['confidence']:.1%}")
```

### Entity Extraction
```python
entities = processor.extract_entities("John Smith entered into an agreement...")
print(entities)  # {'PERSON': ['John Smith'], 'ACTION': ['agreement']}
```

### Document Similarity
```python
similarity = processor.calculate_similarity(doc1, doc2)
print(f"Similarity: {similarity:.2f}")  # 0-1 scale
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 📧 Support

- 📖 [Full Documentation](https://github.com/YOUR_USERNAME/legal-bert-nlp/wiki)
- 🐛 [Report Issues](https://github.com/YOUR_USERNAME/legal-bert-nlp/issues)
- 💬 [Discussions](https://github.com/YOUR_USERNAME/legal-bert-nlp/discussions)

---

**Last Updated:** April 2026
**Status:** Production Ready ✅
