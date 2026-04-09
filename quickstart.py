#!/usr/bin/env python
"""
QUICK START GUIDE - Legal BERT NLP
Run this script to get started immediately!
"""
import os
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def check_installation():
    """Check if all dependencies are installed"""
    print_header("1. CHECKING INSTALLATION")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "sklearn",
        "pandas",
        "numpy",
        "streamlit",
        "sentence_transformers"
    ]
    
    missing = []
    installed_versions = {}
    
    for package in required_packages:
        try:
            if package == "sklearn":
                import sklearn
                installed_versions[package] = sklearn.__version__
            elif package == "transformers":
                import transformers
                installed_versions[package] = transformers.__version__
            else:
                __import__(package)
                mod = sys.modules[package]
                if hasattr(mod, '__version__'):
                    installed_versions[package] = mod.__version__
                else:
                    installed_versions[package] = "✓"
        except ImportError:
            missing.append(package)
    
    # Display results
    print("✅ Installed packages:")
    for pkg, version in installed_versions.items():
        print(f"   • {pkg:<20} v{version}")
    
    if missing:
        print(f"\n❌ Missing packages:")
        for pkg in missing:
            print(f"   • {pkg}")
        print(f"\n💡 Install with: pip install -r requirements.txt")
        return False
    
    # Check GPU
    print(f"\n🔧 GPU Status:")
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available: {device_name}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print(f"   ℹ️  CUDA not available (using CPU)")
            print(f"   💡 Install CUDA for 10x speedup")
    except Exception as e:
        print(f"   ⚠️  Could not check GPU: {e}")
    
    return True


def test_models():
    """Test basic model loading"""
    print_header("2. TESTING MODELS")
    
    print("Loading pretrained models...")
    print("  This may take a few minutes on first run\n")
    
    try:
        print("📦 Loading classification model...")
        from models.classification_model import LegalDocumentClassifier
        classifier = LegalDocumentClassifier(device="cpu")
        print("   ✅ Classification model loaded")
        
        # Test prediction
        test_text = "This is a legal contract."
        label, conf = classifier.predict_single(test_text, return_proba=True)
        print(f"   ✅ Test classification: {label} ({conf:.1%})")
        
    except Exception as e:
        print(f"   ❌ Error loading classification model: {e}")
        return False
    
    try:
        print("\n📦 Loading NER model...")
        from models.ner_model import LegalEntityRecognizer
        ner = LegalEntityRecognizer(device="cpu")
        print("   ✅ NER model loaded")
        
    except Exception as e:
        print(f"   ⚠️  Warning loading NER model: {e}")
    
    try:
        print("\n📦 Loading similarity model...")
        from models.similarity_model import LegalSemanticSimilarity
        sim = LegalSemanticSimilarity(device="cpu")
        print("   ✅ Similarity model loaded")
        
    except Exception as e:
        print(f"   ⚠️  Warning loading similarity model: {e}")
    
    return True


def show_quick_examples():
    """Show quick examples"""
    print_header("3. QUICK EXAMPLES")
    
    print("Example 1: Classification")
    print("-" * 50)
    
    try:
        from models.classification_model import LegalDocumentClassifier
        
        classifier = LegalDocumentClassifier(device="cpu")
        docs = [
            "Agreement between parties for services",
            "The court ruled in favor of the defendant",
            "Section 101 states the requirement",
            "Plaintiff appeals the decision"
        ]
        
        print("\n📄 Classifying documents:\n")
        for doc in docs:
            label, conf = classifier.predict_single(doc, return_proba=True)
            emoji = "📋" if label == "contract" else "⚖️" if label == "case" else "📜"
            print(f"  {emoji} {doc[:40]}...")
            print(f"      → {label.upper()} ({conf:.1%})\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def show_menu():
    """Show interactive menu"""
    print_header("4. NEXT STEPS")
    
    options = [
        ("Run Streamlit UI", "streamlit run app/streamlit_app.py"),
        ("Run All Examples", "python examples.py"),
        ("Train Custom Model", "python -c \"from examples import example_3_training; example_3_training()\""),
        ("Read Full README", "cat README.md"),
        ("Exit", "")
    ]
    
    print("Choose what to do next:\n")
    for i, (desc, cmd) in enumerate(options, 1):
        print(f"  {i}. {desc}")
    
    print("\n" + "="*70)
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Launching Streamlit UI...\n")
            os.system("streamlit run app/streamlit_app.py")
        
        elif choice == "2":
            print("\n▶️  Running examples...\n")
            os.system("python examples.py")
        
        elif choice == "3":
            print("\n🏋️ Starting training...\n")
            from examples import example_3_training
            example_3_training()
        
        elif choice == "4":
            print("\n📖 Opening README...\n")
            os.system("cat README.md" if os.name != "nt" else "type README.md")
        
        elif choice == "5":
            print("\n👋 Goodbye!")
        
        else:
            print("\n❓ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def show_documentation():
    """Show useful documentation"""
    print_header("DOCUMENTATION & RESOURCES")
    
    docs = """
    📚 QUICK REFERENCE:
    
    1. BASIC USAGE:
       from models.classification_model import LegalDocumentClassifier
       classifier = LegalDocumentClassifier()
       label, confidence = classifier.predict_single(text, return_proba=True)
    
    2. FULL PIPELINE:
       from inference.processor import LegalDocumentProcessor
       processor = LegalDocumentProcessor()
       result = processor.process_document(text)
    
    3. SIMILARITY:
       from models.similarity_model import LegalSemanticSimilarity
       similarity = LegalSemanticSimilarity()
       score = similarity.similarity(doc1, doc2)
    
    4. BATCH PROCESSING:
       from inference.processor import batch_process_documents
       results = batch_process_documents(documents, processor)
    
    5. CUSTOM TRAINING:
       from training.train_classifier import TrainingPipeline
       pipeline = TrainingPipeline()
       classifier, history = pipeline.train(train_texts, train_labels, ...)
    
    📖 FOR MORE INFO:
    - Run: streamlit run app/streamlit_app.py (interactive UI)
    - Check: examples.py (6 complete examples)
    - Read: README.md (full documentation)
    
    🔗 EXTERNAL RESOURCES:
    - Legal BERT: https://huggingface.co/nlpaueb/legal-bert-base-uncased
    - HuggingFace: https://huggingface.co/docs/transformers/
    - PyTorch: https://pytorch.org/docs/stable/index.html
    
    💡 TIPS:
    - First run will download models (~500MB) - be patient!
    - Use GPU for 10x faster inference: device="cuda"
    - For long documents, use processor.chunk_document()
    - Check config.py for hyperparameters
    """
    
    print(docs)


def main():
    """Main menu"""
    os.system("cls" if os.name == "nt" else "clear")
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "⚖️  LEGAL BERT NLP - QUICK START" + " "*21 + "║")
    print("║" + " "*12 + "Advanced NLP for Legal Documents" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    # Step 1: Check installation
    if not check_installation():
        print("\n❌ Please install missing dependencies first!")
        return
    
    # Step 2: Test models
    if not test_models():
        print("\n⚠️  Some models failed to load, but you can still proceed")
    
    # Step 3: Quick examples
    show_quick_examples()
    
    # Step 4: Show menu
    show_documentation()
    show_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
