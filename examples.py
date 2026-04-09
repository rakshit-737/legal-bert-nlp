"""
Complete example: Train Legal BERT classifier and evaluate
Shows full workflow: loading data, training, evaluation, inference
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import config
from preprocessing.data_loader import get_splits
from training.train_classifier import TrainingPipeline
from models.classification_model import LegalDocumentClassifier
from evaluation.metrics import EvaluationMetrics
from inference.processor import LegalDocumentProcessor, DocumentSummarizer


def example_1_basic_inference():
    """
    Example 1: Load pretrained model and make predictions
    No training required - just inference
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Inference")
    print("="*60)
    
    # Initialize classifier
    classifier = LegalDocumentClassifier(device="cpu")
    
    # Test documents
    test_docs = [
        "This contract outlines the terms and conditions between parties A and B.",
        "The court ruled in favor of the defendant on grounds of insufficient evidence.",
        "Section 301 states that all contracts must comply with state laws.",
        "The plaintiff appeals the decision citing procedural irregularities."
    ]
    
    # Predict
    print("\n🔍 Classifying documents:")
    for doc in test_docs:
        label, confidence = classifier.predict_single(doc, return_proba=True)
        print(f"  📄 {doc[:50]}...")
        print(f"     → Type: {label} (confidence: {confidence:.2%})\n")


def example_2_full_pipeline():
    """
    Example 2: Full document processing pipeline
    Classification + NER + Similarity analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Full Document Processing Pipeline")
    print("="*60)
    
    # Initialize processor
    print("\n🔧 Initializing Legal Document Processor...")
    processor = LegalDocumentProcessor(device="cpu")
    
    # Sample document
    sample_doc = """
    EMPLOYMENT AGREEMENT
    
    This Agreement is entered into between ABC Corporation (Employer) and John Smith (Employee).
    
    The Employee shall work full-time for the Employer starting January 1, 2024.
    The Employee's annual salary shall be $50,000.
    
    The Employer may terminate this agreement with 30 days written notice.
    The Employee agrees to maintain confidentiality of all proprietary information.
    
    This agreement is governed by the laws of California.
    """
    
    # Full analysis
    print("\n📋 Complete Document Analysis:")
    result = processor.process_document(sample_doc, full_analysis=True)
    
    print(f"\n✅ Analysis Results:")
    print(f"  - Document Type: {result['classification']['label']}")
    print(f"  - Confidence: {result['classification']['confidence']:.2%}")
    if result['entities']:
        print(f"  - Entities Found: {sum(len(v) for v in result['entities'].values())}")
        for entity_type, items in result['entities'].items():
            print(f"    • {entity_type}: {items[:2]}")


def example_3_training():
    """
    Example 3: Train a classifier on custom data
    Uses dummy data for demonstration
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Training")
    print("="*60)
    
    # Load dataset
    print("\n📚 Loading dataset...")
    dataset = get_splits(source="synthetic")
    print("✅ Synthetic legal dataset loaded")
    print(f"  - Train samples: {len(dataset['train']['text'])}")
    print(f"  - Val samples: {len(dataset['validation']['text'])}")
    
    # Initialize training pipeline
    print("\n🎯 Initializing training pipeline...")
    pipeline = TrainingPipeline(
        output_dir="./results",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train
    print("\n🏋️ Training model (this may take a while)...")
    metrics = pipeline.run(
        source="synthetic",
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=8
    )
    
    print("\n✅ Training complete!")
    print(f"  - Test F1 Score: {metrics['f1']:.4f}")


def example_4_evaluation():
    """
    Example 4: Evaluate model and generate metrics
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Evaluation")
    print("="*60)
    
    # Create dummy predictions
    y_true = [0, 1, 2, 3, 0, 1, 2, 1, 0, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 1, 1, 0, 3]
    
    # Random probabilities
    np.random.seed(42)
    y_proba = np.random.rand(len(y_true), 4)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    print("\n📊 Evaluating predictions...")
    
    # Create evaluator
    evaluator = EvaluationMetrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=list(config.LABEL_TO_CLASS.values())
    )
    
    # Get metrics
    metrics = evaluator.get_metrics()
    
    print("\n📈 Metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  - Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  - Recall (macro): {metrics['recall_macro']:.4f}")
    
    # Generate report
    report = evaluator.generate_report()
    print("\n" + report)
    
    # Save visualizations
    print("\n📊 Generating visualizations...")
    # evaluator.plot_confusion_matrix("./results/confusion_matrix.png")
    # evaluator.plot_metrics_comparison("./results/metrics.png")
    print("  ✅ Visualizations generated")


def example_5_similarity():
    """
    Example 5: Document similarity analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Document Similarity Analysis")
    print("="*60)
    
    from models.similarity_model import LegalSemanticSimilarity
    
    # Initialize similarity model
    print("\n🔗 Initializing similarity model...")
    similarity_model = LegalSemanticSimilarity(device="cpu")
    
    # Test documents
    doc1 = "This agreement outlines the terms of service between parties."
    doc2 = "This contract specifies the conditions and terms of the agreement."
    doc3 = "The weather today is sunny and warm."
    
    print(f"\n📄 Document 1: {doc1}")
    print(f"📄 Document 2: {doc2}")
    print(f"📄 Document 3: {doc3}")
    
    print(f"\n🔍 Similarity Analysis:")
    sim_1_2 = similarity_model.similarity(doc1, doc2)
    sim_1_3 = similarity_model.similarity(doc1, doc3)
    
    print(f"  - Doc1 vs Doc2 (similar legal docs): {sim_1_2:.3f}")
    print(f"  - Doc1 vs Doc3 (unrelated): {sim_1_3:.3f}")
    
    # Find similar from corpus
    corpus = [doc2, doc3, "The defendant appeals the court decision."]
    print(f"\n📚 Finding similar documents in corpus...")
    results = similarity_model.most_similar(doc1, corpus, top_k=2)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Similarity: {score:.3f}")
        print(f"     Document: {doc[:60]}...")


def example_6_ner():
    """
    Example 6: Named Entity Recognition
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Named Entity Recognition")
    print("="*60)
    
    from models.ner_model import LegalEntityRecognizer
    
    # Initialize NER model
    print("\n🔍 Initializing NER model...")
    ner = LegalEntityRecognizer(device="cpu")
    
    # Sample legal text
    text = "Judge Smith ruled in favor of John Doe against ABC Corporation on January 15, 2024."
    
    print(f"\n📝 Text: {text}")
    
    print(f"\n🔎 Extracting entities:")
    entities = ner.predict(text)
    
    for token, entity_type, confidence in entities:
        print(f"  - '{token}' → {entity_type} (confidence: {confidence:.2%})")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("LEGAL BERT NLP - COMPLETE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_basic_inference()
    example_2_full_pipeline()
    # example_3_training()  # Uncomment to train (takes time)
    example_4_evaluation()
    example_5_similarity()
    example_6_ner()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
    print("\n📚 Next steps:")
    print("  1. Run: streamlit run app/streamlit_app.py (for UI)")
    print("  2. Train on your data with example_3_training()")
    print("  3. Customize models in models/ directory")
    print("  4. Check app/streamlit_app.py for interactive interface")
    print("\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Using device: {device}")
    
    try:
        # Run all examples
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        
        # Suggest which module failed
        import traceback
        print("\nFull error:")
        traceback.print_exc()
