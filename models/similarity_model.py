"""
Semantic Similarity Model for Legal Documents
Measures similarity between legal terms, clauses, and documents
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LegalSemanticSimilarity:
    """
    Semantic similarity using sentence embeddings
    Compares legal documents, clauses, and terms
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize semantic similarity model
        Recommend: "all-MiniLM-L6-v2" or "all-mpnet-base-v2" for legal texts
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings_cache = {}
    
    def get_embedding(self, text: str, cache: bool = True) -> np.ndarray:
        """Get embedding for text"""
        if cache and text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        
        if cache:
            self.embeddings_cache[text] = embedding
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts"""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (0-1)
        """
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def most_similar(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar documents to query
        Returns: [(document, similarity_score), ...]
        """
        query_embedding = self.get_embedding(query)
        doc_embeddings = self.get_embeddings_batch(documents)
        
        similarities = []
        for idx, doc_embedding in enumerate(doc_embeddings):
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((documents[idx], float(sim)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Create similarity matrix for multiple texts
        """
        embeddings = self.get_embeddings_batch(texts)
        similarities = np.zeros((len(texts), len(texts)))
        
        for i in range(len(texts)):
            for j in range(len(texts)):
                similarities[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
        
        return similarities
    
    def cluster_documents(self, documents: List[str], threshold: float = 0.7) -> List[List[int]]:
        """
        Cluster documents by semantic similarity
        Returns: [[doc_indices], ...] groups of similar documents
        """
        sim_matrix = self.similarity_matrix(documents)
        # Clustering using connected components (handles transitive similarity)
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        clusters = []
        visited = set()
        
        # Use scipy for efficient connected component clustering
        try:
            # Create adjacency matrix from similarity
            adjacency_np = (sim_matrix >= threshold).astype(float)
            
            # Find connected components
            graph = csr_matrix(adjacency_np)
            n_components, labels = connected_components(csgraph=graph, directed=False)
            
            # Group indices by component
            for component_id in range(n_components):
                cluster = [i for i, label in enumerate(labels) if label == component_id]
                if len(cluster) > 0:
                    clusters.append(cluster)
        except Exception:
            # Fallback to simple threshold-based clustering
            for i in range(len(documents)):
                if i in visited:
                    continue
                
                cluster = [i]
                visited.add(i)
                
                for j in range(i + 1, len(documents)):
                    if sim_matrix[i, j] >= threshold:
                        cluster.append(j)
                        visited.add(j)
                
                clusters.append(cluster)
        
        return clusters
    
    def find_duplicate_clauses(self, clauses: List[str], threshold: float = 0.85) -> List[Tuple[int, int, float]]:
        """
        Find potentially duplicate or very similar clauses
        Returns: [(idx1, idx2, similarity), ...]
        """
        duplicates = []
        sim_matrix = self.similarity_matrix(clauses)
        
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                if sim_matrix[i, j] >= threshold:
                    duplicates.append((i, j, sim_matrix[i, j]))
        
        return duplicates
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embeddings_cache.clear()


class BERTSemanticSimilarity:
    """
    Semantic similarity using BERT embeddings
    Fine-tuned for legal documents
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading BERT semantic model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get [CLS] token embedding from BERT"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return cls_embedding.cpu()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return float(similarity)
