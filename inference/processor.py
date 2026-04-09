"""
Inference and application module
Complete pipeline for document processing
"""
import torch
import sys
import os
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.classification_model import LegalDocumentClassifier
from models.ner_model import LegalEntityRecognizer
from models.similarity_model import LegalSemanticSimilarity
from preprocessing.text_cleaner import TextCleaner, DocumentPreprocessor


class LegalDocumentProcessor:
    """
    Complete end-to-end legal document processing pipeline
    Includes: text cleaning, classification, NER, similarity
    """
    
    def __init__(self, classification_model_path: str = None,
                 ner_model_path: str = None,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_cleaner = TextCleaner()
        self.doc_preprocessor = DocumentPreprocessor()
        
        print(f"🚀 Initializing Legal Document Processor on {self.device}")
        
        # Load models
        self.classifier = LegalDocumentClassifier(
            model_name=classification_model_path or config.DEFAULT_MODEL,
            device=self.device
        )
        
        self.ner = LegalEntityRecognizer(
            model_name=ner_model_path or config.DEFAULT_MODEL,
            device=self.device
        )
        
        self.similarity = LegalSemanticSimilarity(device=self.device)
        
        print("✅ All models loaded successfully!")
    
    def clean_document(self, text: str) -> str:
        """Clean raw legal document text"""
        cleaned = self.text_cleaner.clean(text)
        return cleaned
    
    def chunk_document(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split long document into chunks"""
        chunks = self.doc_preprocessor.truncate_text(text, max_length=chunk_size)
        return chunks
    
    def classify_document(self, text: str, return_proba: bool = True) -> Dict:
        """
        Classify document type
        Returns: {label: str, confidence: float}
        """
        text = self.clean_document(text)
        
        if return_proba:
            pred, proba = self.classifier.predict([text], return_probabilities=True)
            return {
                "label": config.LABEL_TO_CLASS[pred[0]],
                "confidence": float(proba[0]),
                "all_scores": {
                    config.LABEL_TO_CLASS[i]: float(score)
                    for i, score in enumerate(self.classifier.predict([text], return_probabilities=True)[1])
                }
            }
        else:
            pred, _ = self.classifier.predict([text])
            return {"label": config.LABEL_TO_CLASS[pred[0]]}
    
    def extract_entities(self, text: str, group_by_type: bool = True) -> Dict:
        """
        Extract named entities
        Returns: {entity_type: [entities], ...} or [(token, type, confidence), ...]
        """
        text = self.clean_document(text)
        
        if group_by_type:
            return self.ner.extract_entities_by_type(text)
        else:
            entities = self.ner.predict(text)
            return {
                "entities": [
                    {"text": token, "type": tag, "confidence": float(conf)}
                    for token, tag, conf in entities
                ]
            }
    
    def find_similar_documents(self, query_doc: str, corpus: List[str],
                              top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar documents from corpus"""
        query_doc = self.clean_document(query_doc)
        corpus_clean = [self.clean_document(doc) for doc in corpus]
        
        return self.similarity.most_similar(query_doc, corpus_clean, top_k)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two documents"""
        text1 = self.clean_document(text1)
        text2 = self.clean_document(text2)
        
        return self.similarity.similarity(text1, text2)
    
    def process_document(self, text: str, full_analysis: bool = True) -> Dict:
        """
        Complete analysis of a document
        """
        print("📄 Processing document...")
        
        # Clean text
        cleaned_text = self.clean_document(text)
        print("✅ Text cleaned")
        
        result = {
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "cleaned_text": cleaned_text
        }
        
        if full_analysis:
            # Classification
            print("🏷️  Classifying document...")
            classification = self.classify_document(cleaned_text)
            result["classification"] = classification
            print(f"   Type: {classification['label']} (confidence: {classification['confidence']:.2%})")
            
            # NER
            print("🔍 Extracting entities...")
            entities = self.extract_entities(cleaned_text, group_by_type=True)
            result["entities"] = entities
            if entities:
                for entity_type, items in entities.items():
                    print(f"   {entity_type}: {items[:3]}...")
            
        return result
    
    def compare_documents(self, doc1: str, doc2: str) -> Dict:
        """Compare two documents for similarity and differences"""
        doc1_clean = self.clean_document(doc1)
        doc2_clean = self.clean_document(doc2)
        
        similarity = self.calculate_similarity(doc1_clean, doc2_clean)
        
        # Extract entities
        entities1 = self.extract_entities(doc1_clean, group_by_type=True)
        entities2 = self.extract_entities(doc2_clean, group_by_type=True)
        
        return {
            "similarity": float(similarity),
            "doc1_entities": entities1,
            "doc2_entities": entities2
        }
    
    def batch_classify(self, documents: List[str]) -> List[Dict]:
        """Classify multiple documents"""
        results = []
        for doc in documents:
            result = self.classify_document(doc)
            results.append(result)
        return results


class DocumentSummarizer:
    """
    Extract key information from legal documents
    """
    
    def __init__(self, processor: LegalDocumentProcessor):
        self.processor = processor
    
    def extract_key_clauses(self, text: str, top_k: int = 5) -> List[str]:
        """Extract key sentences/clauses"""
        sentences = text.split(". ")
        
        if len(sentences) <= top_k:
            return sentences
        
        # Score sentences by keyword presence
        scores = []
        legal_keywords = [
            "agreement", "clause", "hereby", "whereas", "obligation",
            "liability", "indemnify", "consent", "default"
        ]
        
        for sentence in sentences:
            score = sum(1 for keyword in legal_keywords
                       if keyword.lower() in sentence.lower())
            scores.append((sentence.strip(), score))
        
        # Return top-k
        sorted_clauses = sorted(scores, key=lambda x: x[1], reverse=True)
        return [clause for clause, _ in sorted_clauses[:top_k]]
    
    def get_document_summary(self, text: str) -> Dict:
        """Generate summary of document"""
        cleaned = self.processor.clean_document(text)
        
        # Get entities
        entities = self.processor.extract_entities(cleaned, group_by_type=True)
        
        # Get classification
        classification = self.processor.classify_document(cleaned)
        
        # Get key clauses
        key_clauses = self.extract_key_clauses(cleaned, top_k=3)
        
        return {
            "type": classification["label"],
            "confidence": classification["confidence"],
            "key_entities": entities,
            "key_clauses": key_clauses,
            "word_count": len(cleaned.split()),
            "estimated_reading_time_minutes": len(cleaned.split()) // 200
        }


def batch_process_documents(documents: List[str], processor: LegalDocumentProcessor,
                           task: str = "classify") -> List[Dict]:
    """
    Process multiple documents in batch
    task: "classify", "extract_entities", "summarize"
    """
    results = []
    
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}...")
        
        if task == "classify":
            result = processor.classify_document(doc)
        elif task == "extract_entities":
            result = processor.extract_entities(doc)
        elif task == "summarize":
            summarizer = DocumentSummarizer(processor)
            result = summarizer.get_document_summary(doc)
        else:
            result = processor.process_document(doc)
        
        results.append(result)
    
    return results
