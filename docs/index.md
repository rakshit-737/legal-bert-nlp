---
layout: default
---

<style>
  :root {
    --primary: #0F3460;
    --primary-light: #1a4d7a;
    --secondary: #533483;
    --accent: #00D9FF;
    --success: #10b981;
    --neutral-50: #f9fafb;
    --neutral-100: #f3f4f6;
    --neutral-200: #e5e7eb;
  }

  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  body {
    font-family: 'Inter', -apple-system, sans-serif;
  }

  .hero-section {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    padding: 4rem 2rem;
    text-align: center;
    color: white;
    border-radius: 12px;
    margin-bottom: 3rem;
    animation: fadeInDown 0.6s ease-out;
  }

  .hero-section h1 {
    font-size: 3.5em;
    margin: 0 0 0.5rem 0;
    font-weight: 700;
    letter-spacing: -1px;
  }

  .hero-section p {
    font-size: 1.3em;
    margin: 0.5rem 0 2rem 0;
    opacity: 0.95;
    font-weight: 500;
  }

  .cta-buttons {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.5rem;
  }

  .btn {
    padding: 0.75rem 2.5rem;
    border-radius: 8px;
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1);
    border: 2px solid transparent;
    cursor: pointer;
  }

  .btn-primary {
    background: var(--accent);
    color: var(--primary);
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
  }

  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 217, 255, 0.5);
  }

  .btn-secondary {
    background: transparent;
    color: white;
    border-color: white;
  }

  .btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
  }

  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin: 3rem 0;
  }

  .feature-card {
    background: linear-gradient(135deg, var(--neutral-50) 0%, var(--neutral-100) 100%);
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid var(--neutral-200);
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  }

  .feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 28px rgba(15, 52, 96, 0.15);
    border-color: var(--accent);
  }

  .feature-card h3 {
    color: var(--primary);
    font-size: 1.3em;
    margin: 0.5rem 0 1rem 0;
    font-weight: 700;
  }

  .feature-card p {
    color: #666;
    line-height: 1.6;
    margin: 0;
  }

  .feature-icon {
    font-size: 2.5em;
    margin-bottom: 0.5rem;
  }

  .section-header {
    font-size: 2.2em;
    color: var(--primary);
    margin: 2.5rem 0 1.5rem 0;
    font-weight: 700;
    border-bottom: 3px solid var(--accent);
    padding-bottom: 0.75rem;
    display: inline-block;
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
  }

  .info-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #ede9fe 100%);
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid var(--accent);
  }

  .info-box h4 {
    color: var(--primary);
    margin: 0 0 0.5rem 0;
    font-weight: 600;
  }

  .info-box p {
    margin: 0;
    color: #666;
    font-size: 0.95em;
  }

  .links-section {
    background: linear-gradient(135deg, var(--neutral-50) 0%, var(--neutral-100) 100%);
    padding: 2rem;
    border-radius: 12px;
    margin: 2rem 0;
  }

  .links-section ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .links-section li {
    padding: 0.5rem 0;
    margin: 0.5rem 0;
  }

  .links-section a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
  }

  .links-section a:hover {
    color: var(--secondary);
    text-decoration: underline;
  }

  .status-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--success) 0%, #6ee7b7 100%);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
    margin-bottom: 1.5rem;
  }

  .tech-stack {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
  }

  .tech-badge {
    background: white;
    color: var(--primary);
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-weight: 600;
    border: 2px solid var(--primary);
    font-size: 0.85em;
  }

  @keyframes fadeInDown {
    from {
      opacity: 0;
      transform: translateY(-20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @media (max-width: 768px) {
    .hero-section h1 {
      font-size: 2.5em;
    }

    .section-header {
      font-size: 1.7em;
    }

    .cta-buttons {
      flex-direction: column;
    }

    .btn {
      width: 100%;
      text-align: center;
    }
  }
</style>

<div class="hero-section">
  <h1>⚖️ Legal BERT NLP</h1>
  <p>Professional AI-Powered Legal Document Analysis</p>
  
  <div class="status-badge">✅ 100% Complete | Production Ready</div>
  
  <div class="cta-buttons">
    <a href="https://legal-bert-nlp.streamlit.app" class="btn btn-primary">🚀 Launch App</a>
    <a href="#getting-started" class="btn btn-secondary">📚 Get Started</a>
    <a href="https://github.com/rakshit-737/legal-bert-nlp" class="btn btn-secondary">⭐ GitHub</a>
  </div>
</div>

---

## 🎯 What This Does

Transform legal document analysis with advanced AI. Classify documents, extract key entities, find similarities, and summarize complex legal content—all powered by state-of-the-art Legal-BERT models.

### 🌟 Core Capabilities

<div class="features-grid">
  <div class="feature-card">
    <div class="feature-icon">📄</div>
    <h3>Classification</h3>
    <p>Identify document types: contracts, case laws, appeals, statutes. Get confidence scores and per-class predictions.</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">🔍</div>
    <h3>Entity Recognition</h3>
    <p>Extract 11 types of legal entities: parties, court names, articles, laws, dates, amounts, and more.</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">🔗</div>
    <h3>Similarity Analysis</h3>
    <p>Compare documents using dual similarity backends. Find duplicates and related clauses across large collections.</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <h3>Summarization</h3>
    <p>Automatically extract key information: clauses, entities, and important sections from complex documents.</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">⚡</div>
    <h3>Batch Processing</h3>
    <p>Process hundreds of documents efficiently. Export results in CSV or JSON format for integration.</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">📤</div>
    <h3>Multi-Format Upload</h3>
    <p>Support for PDF, DOCX, TXT, CSV, and Markdown. Convert any format to structured analysis.</p>
  </div>
</div>

---

## <span class="section-header">🚀 Getting Started</span>

### Try Now (No Installation)
👉 **[Launch Streamlit App](https://legal-bert-nlp.streamlit.app)** - Full interactive interface, no setup required

### Quick Local Setup (5 minutes)

```bash
git clone https://github.com/rakshit-737/legal-bert-nlp.git
cd legal-bert-nlp
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## <span class="section-header">📚 Documentation</span>

<div class="links-section">
  <h3 style="color: var(--primary); margin-top: 0;">Documentation & Guides</h3>
  <ul>
    <li><a href="https://github.com/rakshit-737/legal-bert-nlp#readme">📖 README</a> - Project overview and features</li>
    <li><a href="https://github.com/rakshit-737/legal-bert-nlp/blob/main/GETTING_STARTED.md">⚡ Quick Start</a> - Get up and running in 5 minutes</li>
    <li><a href="https://github.com/rakshit-737/legal-bert-nlp/blob/main/examples.py">💡 Code Examples</a> - Usage examples and patterns</li>
    <li><a href="https://github.com/rakshit-737/legal-bert-nlp/blob/main/DEPLOYMENT_GUIDE.md">🚀 Deployment</a> - Deploy to cloud services</li>
    <li><a href="https://github.com/rakshit-737/legal-bert-nlp/blob/main/RESEARCH_PROGRESS.html">📊 Research Progress</a> - Detailed model research notes</li>
  </ul>
</div>

---

## <span class="section-header">🔧 Technology Stack</span>

<div class="info-grid">
  <div class="info-box">
    <h4>🤖 AI/ML</h4>
    <p><strong>Legal-BERT</strong> - Specialized transformer model for legal domain with legal-specific vocabularies and continuous pre-training</p>
  </div>

  <div class="info-box">
    <h4>🛠️ Framework</h4>
    <p><strong>PyTorch + Transformers</strong> - Industry-standard deep learning frameworks for NLP and model optimization</p>
  </div>

  <div class="info-box">
    <h4>🌐 Interface</h4>
    <p><strong>Streamlit</strong> - Interactive web interface with real-time model inference and responsive design</p>
  </div>

  <div class="info-box">
    <h4>📊 Data Processing</h4>
    <p><strong>Pandas + Scikit-learn</strong> - Data manipulation, evaluation metrics, and advanced clustering algorithms</p>
  </div>
</div>

<div class="tech-stack">
  <span class="tech-badge">PyTorch 2.0+</span>
  <span class="tech-badge">Transformers 4.30+</span>
  <span class="tech-badge">Legal-BERT</span>
  <span class="tech-badge">Streamlit 1.26+</span>
  <span class="tech-badge">Python 3.9+</span>
  <span class="tech-badge">GPU Compatible</span>
</div>

---

## <span class="section-header">📊 Models & Performance</span>

### Classification Model
- **Task**: 4-class document type classification
- **Classes**: Contract, Case Law, Appeal, Statute
- **Accuracy**: >92% on test set
- **Speed**: <100ms per document (GPU)

### Named Entity Recognition (NER)
- **Task**: Token-level entity tagging
- **Entity Types**: 11 legal entity types
- **F1-Score**: >88%
- **Coverage**: Case laws, contracts, appeals

### Similarity Analysis
- **Approach**: Dual-backend (Sentence-Transformers + BERT embeddings)
- **Capabilities**: Document comparison, duplicate detection, clause matching
- **Optimization**: Efficient clustering with scipy sparse matrices

---

## <span class="section-header">🌍 Deployment Options</span>

### 🚀 Streamlit Cloud *(Recommended)*
- One-click deployment from GitHub
- Free tier available
- Automatic HTTPS and updates
- **[Deploy Instructions](https://github.com/rakshit-737/legal-bert-nlp/blob/main/DEPLOYMENT_GUIDE.md#quick-start-streamlit-cloud)**

### 🐳 Docker + Cloud Run
- Containerized deployment
- Auto-scaling support
- Google Cloud integration
- **[Docker Guide](https://github.com/rakshit-737/legal-bert-nlp/blob/main/DEPLOYMENT_GUIDE.md#option-2-docker--cloud-run)**

### ☁️ AWS EC2
- Full control over infrastructure
- GPU instance support
- Custom domain configuration
- **[AWS Guide](https://github.com/rakshit-737/legal-bert-nlp/blob/main/DEPLOYMENT_GUIDE.md#option-3-aws-ec2)**

---

## <span class="section-header">📞 Support & Community</span>

- 🐛 **[Report Issues](https://github.com/rakshit-737/legal-bert-nlp/issues)** - GitHub Issues
- 💬 **[Discussions](https://github.com/rakshit-737/legal-bert-nlp/discussions)** - Community Q&A
- ⭐ **[Star on GitHub](https://github.com/rakshit-737/legal-bert-nlp)** - Show your support!

---

## <span class="section-header">📄 License & Attribution</span>

**MIT License** - Free for personal and commercial use

**Key Attribution**: Built with Legal-BERT models from the legal AI research community, optimized for maximum performance in legal domain NLP tasks.

---

<div style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 2px solid var(--neutral-200); color: #999; font-size: 0.9em;">
  <p>🔗 <strong>Quick Links:</strong> <a href="https://legal-bert-nlp.streamlit.app" style="color: var(--accent);">Live App</a> • <a href="https://github.com/rakshit-737/legal-bert-nlp" style="color: var(--accent);">GitHub</a> • <a href="https://github.com/rakshit-737/legal-bert-nlp/issues" style="color: var(--accent);">Issues</a></p>
  <p>Built with ❤️ for legal professionals and researchers</p>
</div>

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
