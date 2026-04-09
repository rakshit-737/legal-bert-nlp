"""
Text cleaning and preprocessing utilities for legal documents
"""
import re
import string
from typing import List, Tuple
import pandas as pd


class TextCleaner:
    """Clean and preprocess legal documents"""
    
    def __init__(self):
        self.punctuation = string.punctuation
        
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters but keep legal symbols"""
        # Keep important symbols like §, ¶, etc.
        text = re.sub(r'[^\w\s§¶\-\.]', '', text)
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()
    
    def remove_headers_footers(self, text: str) -> str:
        """Remove page headers, footers, page numbers"""
        # Remove page numbers
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-\s*\d+\s*-', '', text)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        return text
    
    def normalize_case(self, text: str, preserve_acronyms: bool = True) -> str:
        """Normalize case (mostly lowercase but preserve certain patterns)"""
        if preserve_acronyms:
            # Keep all-caps words (likely acronyms or important)
            words = text.split()
            normalized = []
            for word in words:
                if len(word) > 1 and word.isupper():
                    normalized.append(word)  # Keep acronyms
                else:
                    normalized.append(word.lower())
            return ' '.join(normalized)
        return text.lower()
    
    def remove_noise(self, text: str) -> str:
        """Remove OCR noise, stamps, etc."""
        # Remove common OCR artifacts
        text = re.sub(r'[^\x20-\x7E\xA0-\xFF]', '', text)
        # Remove repeated characters (likely noise)
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        return text
    
    def clean(self, text: str, remove_noise: bool = True) -> str:
        """Full cleaning pipeline"""
        if not text:
            return ""
        
        text = self.remove_headers_footers(text)
        if remove_noise:
            text = self.remove_noise(text)
        text = self.normalize_case(text)
        text = self.remove_extra_whitespace(text)
        
        return text


class DocumentPreprocessor:
    """Preprocess documents for BERT"""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split document into sentences"""
        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Truncate long text into chunks to fit BERT's max length
        With overlap for context preservation
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def extract_metadata(text: str) -> dict:
        """Extract metadata from legal document"""
        metadata = {
            "has_section_numbers": bool(re.search(r'^§\s*\d+', text, re.MULTILINE)),
            "has_dates": bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', text)),
            "has_case_number": bool(re.search(r'Case\s+No\.?\s*[A-Z0-9\-]+', text, re.IGNORECASE)),
            "word_count": len(text.split()),
            "char_count": len(text),
        }
        return metadata


class PDFProcessor:
    """Extract and process text from PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("pdfplumber not installed. Install with: pip install pdfplumber")
            return ""
    
    @staticmethod
    def extract_text_with_ocr(pdf_path: str) -> str:
        """Extract text from PDF using OCR (pytesseract)"""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            return text
        except ImportError:
            print("pytesseract or pdf2image not installed")
            return ""


def preprocess_dataset(df: pd.DataFrame, 
                       text_column: str = "text",
                       label_column: str = "label") -> pd.DataFrame:
    """Preprocess entire dataset"""
    cleaner = TextCleaner()
    preprocessor = DocumentPreprocessor()
    
    # Clean text
    df[f"{text_column}_cleaned"] = df[text_column].apply(cleaner.clean)
    
    # Extract metadata
    metadata = df[f"{text_column}_cleaned"].apply(preprocessor.extract_metadata)
    for col in metadata.iloc[0].keys():
        df[f"{col}"] = metadata.apply(lambda x: x[col])
    
    return df
