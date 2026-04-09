"""
inference/processor.py
======================
Inference wrapper for the trained OptiBERT model.

Supports:
  • Single-document classification
  • Batch classification
  • Named Entity Recognition (rule-based + BERT-assisted)
  • Semantic similarity between legal documents
  • Document summarisation (extractive, sentence-scored)
"""

from __future__ import annotations
import re
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────────────────────────
# NER patterns  (legal entity rules aligned with paper Section IV-B)
# ─────────────────────────────────────────────────────────────────────────────

_NER_PATTERNS: Dict[str, str] = {
    "DATE":     r"\b(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
                r"|\b\d{1,2}/\d{1,2}/\d{4}\b",
    "CASE_NO":  r"\b\d{4}-[A-Z]{2}-\d{4,6}\b",
    "STATUTE":  r"\b(?:Section|§|Art\.?)\s*\d+[\w.]*\b",
    "ORG":      r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
                r"(?:Corp(?:oration)?|LLC|Inc(?:orporated)?|Ltd|"
                r"LLP|Foundation|Authority|Commission|Bureau|Department)\b",
    "JUDGE":    r"\b(?:Judge|Justice|Hon\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
    "PERSON":   r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
    "CLAUSE":   r"\b(?:Section|Clause|Article|Paragraph|Schedule|Exhibit)\s+[A-Z0-9]+\b",
    "AMOUNT":   r"\$\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?\b",
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Rule-based NER for legal documents."""
    found: Dict[str, List[str]] = {}
    for entity_type, pattern in _NER_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Deduplicate while preserving order
        seen: set = set()
        unique = [m for m in matches if not (m in seen or seen.add(m))]
        if unique:
            found[entity_type] = unique
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Legal Document Processor
# ─────────────────────────────────────────────────────────────────────────────

class LegalDocumentProcessor:
    """
    End-to-end processor for legal documents.
    Loads the trained OptiBERT checkpoint automatically.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._classifier = None
        self._sentence_model = None
        self.model_path = model_path or str(Path(config.CHECKPOINTS) / "best_optibert.pt")
        self._load_classifier()

    def _load_classifier(self):
        """Lazy-load the classifier (downloads model on first call)."""
        try:
            from models.classification_model import LegalDocumentClassifier
            self._classifier = LegalDocumentClassifier(
                model_name=config.DEFAULT_MODEL,
                device=self.device,
                use_custom=True,
            )
            model = self._classifier.get_model()

            ckpt = Path(self.model_path)
            if ckpt.exists():
                print(f"✅ Loading checkpoint: {ckpt}")
                model.load_state_dict(
                    torch.load(str(ckpt), map_location=self.device)
                )
            else:
                print(f"⚠️  No checkpoint at {ckpt}. Using pre-trained weights only.")
        except Exception as exc:
            print(f"⚠️  Classifier load error: {exc}")

    # ── classification ────────────────────────────────────────────────────────

    def classify(self, text: str) -> Dict:
        """
        Classify a single legal document.

        Returns:
            {label, confidence, all_scores}
        """
        if self._classifier is None:
            return {"label": "unknown", "confidence": 0.0, "all_scores": {}}

        self._classifier.get_model().eval()
        with torch.no_grad():
            inputs = self._classifier.preprocess([text])
            model  = self._classifier.get_model()
            logits = model(**inputs)
            if hasattr(logits, "logits"):
                logits = logits.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx  = int(probs.argmax())
        label     = config.LABEL_TO_CLASS.get(pred_idx, "unknown")
        all_scores = {
            config.LABEL_TO_CLASS[i]: float(probs[i])
            for i in range(len(probs))
        }
        return {
            "label":      label,
            "confidence": float(probs[pred_idx]),
            "all_scores": all_scores,
        }

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a list of documents."""
        return [self.classify(t) for t in texts]

    # ── NER ───────────────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from legal text."""
        return extract_entities(text)

    # ── similarity ────────────────────────────────────────────────────────────

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Cosine similarity between two legal documents.
        Uses sentence-transformers if available, else falls back to TF-IDF cosine.
        """
        try:
            from sentence_transformers import SentenceTransformer
            if self._sentence_model is None:
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = self._sentence_model.encode([text1, text2], convert_to_tensor=True)
            cos  = torch.nn.functional.cosine_similarity(embs[0:1], embs[1:2]).item()
            return float(cos)
        except ImportError:
            return self._tfidf_similarity(text1, text2)

    @staticmethod
    def _tfidf_similarity(t1: str, t2: str) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        vect = TfidfVectorizer().fit([t1, t2])
        mat  = vect.transform([t1, t2])
        return float(sk_cos(mat[0], mat[1])[0, 0])

    # ── extractive summarisation ──────────────────────────────────────────────

    @staticmethod
    def summarise(text: str, num_sentences: int = 3) -> str:
        """
        Extractive summarisation using TF-IDF sentence scoring.
        Selects the top-N most informative sentences.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 5]
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect   = TfidfVectorizer(stop_words="english")
            matrix = vect.fit_transform(sentences)
            scores = matrix.sum(axis=1).A1
            top    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
            top.sort()
            return " ".join(sentences[i] for i in top)
        except ImportError:
            return " ".join(sentences[:num_sentences])

    # ── full analysis ─────────────────────────────────────────────────────────

    def analyse(self, text: str) -> Dict:
        """
        Run full deep-contextual analysis on a document:
          1. Classification
          2. NER
          3. Extractive summary
        """
        return {
            "classification": self.classify(text),
            "entities":       self.extract_entities(text),
            "summary":        self.summarise(text),
            "word_count":     len(text.split()),
            "char_count":     len(text),
        }
