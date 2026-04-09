"""
Domain-Specific Enhancements for Legal BERT

Research-backed improvements specifically designed for legal document processing:
- Legal vocabulary expansion
- Domain-specific attention masking
- Legal entity aware contrastive learning
- Clause-level analysis
- Document structure preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re


class LegalVocabularyEnhancer:
    """
    Expand and enhance vocabulary for legal-specific terms.
    
    Research shows domain-specific vocabulary significantly improves
    performance on specialized documents (legal, medical, etc.)
    """
    
    # Common legal terms and their variations
    LEGAL_TERMS = {
        'defendant': ['respondent', 'accused', 'perpetrator'],
        'plaintiff': ['applicant', 'claimant', 'petitioner'],
        'contract': ['agreement', 'accord', 'covenant'],
        'clause': ['provision', 'section', 'article', 'paragraph'],
        'liability': ['responsibility', 'obligation', 'duty'],
        'damages': ['compensation', 'reparation', 'indemnity'],
        'jurisdiction': ['authority', 'power', 'competence'],
        'statute': ['law', 'legislation', 'ordinance', 'regulation'],
    }
    
    # Legal abbreviations
    LEGAL_ABBREVIATIONS = {
        'et al.': 'and others',
        'i.e.': 'that is',
        'e.g.': 'for example',
        'vs': 'versus',
        'v.': 'versus',
        'Inc.': 'Incorporated',
        'LLC': 'Limited Liability Company',
        'Corp': 'Corporation',
        'Ltd': 'Limited',
    }
    
    @staticmethod
    def standardize_legal_terms(text: str) -> str:
        """Standardize legal terms for consistency."""
        for abbr, expansion in LegalVocabularyEnhancer.LEGAL_ABBREVIATIONS.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def get_legal_entity_pattern() -> str:
        """Get regex pattern for common legal entities."""
        pattern = r'\b([A-Z][a-z]+ v\.? [A-Z][a-z]+|' \
                 r'[A-Z][a-z]+ et al\.? v\.? [A-Z][a-z]+|' \
                 r'[A-Z][a-z]+ Corporation|' \
                 r'[A-Z][a-z]+ LLC|' \
                 r'[A-Z][a-z]+ Inc\.?)\b'
        return pattern
    
    @staticmethod
    def extract_case_names(text: str) -> List[str]:
        """Extract case names from legal text."""
        pattern = LegalVocabularyEnhancer.get_legal_entity_pattern()
        matches = re.findall(pattern, text)
        return matches


class DomainSpecificAttention(nn.Module):
    """
    Attention mechanism with domain-specific masking for legal documents.
    
    Masks irrelevant tokens (dates, citation numbers) to focus on legal content.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.hidden_size = hidden_size
    
    @staticmethod
    def create_legal_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        """
        Create attention mask highlighting legal content tokens.
        
        Tokens to mask (reduce attention weight):
        - Citation numbers (e.g., 123 F.3d 456)
        - Section numbers (e.g., § 123.456)
        - Page numbers (e.g., p. 123)
        - URLs and emails
        - Punctuation-heavy sequences
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        mask = torch.ones((batch_size, seq_length), dtype=torch.float32, device=device)
        
        # Decode tokens and apply pattern-based masking
        try:
            for batch_idx in range(batch_size):
                tokens = tokenizer.decode(input_ids[batch_idx], skip_special_tokens=False)
                token_strs = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
                
                for token_idx, token_str in enumerate(token_strs):
                    # Reduce attention on citations (e.g., "123", "F.3d", "U.S.")
                    if re.match(r'^\d+$', token_str) or re.match(r'^[FUSCR]+\.\d+$', token_str):
                        mask[batch_idx, token_idx] = 0.7
                    
                    # Reduce attention on section markers
                    elif token_str in ['§', 'section', 'article', 'clause']:
                        mask[batch_idx, token_idx] = 0.8
                    
                    # Reduce attention on page references
                    elif re.match(r'^(p|pp)\.?$', token_str.lower()) or re.match(r'^\d+$', token_str):
                        mask[batch_idx, token_idx] = 0.7
                    
                    # Reduce attention on URLs and emails
                    elif re.match(r'^(https?:|www\.|.*@)', token_str):
                        mask[batch_idx, token_idx] = 0.5
                    
                    # Slightly reduce attention on pure punctuation
                    elif re.match(r'^[\p{P}]+$', token_str):
                        mask[batch_idx, token_idx] = 0.9
        except Exception as e:
            # If masking fails, return default mask
            pass
        
        return mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                domain_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with domain-specific attention masking.
        """
        attn_output, attn_weights = self.attention(query, key, value, average_attn_weights=False)
        
        if domain_mask is not None:
            # Apply domain mask to attention weights
            attn_weights = attn_weights * domain_mask.unsqueeze(1)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        
        return attn_output, attn_weights


class ContrastiveLearningLoss(nn.Module):
    """
    Contrastive loss for learning legal document similarity.
    
    Pulls together documents of same type, pushes apart different types.
    (SimCLR-inspired, adapted for document classification)
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Document embeddings [batch_size, embedding_dim]
            labels: Document class labels [batch_size]
        
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive and negative masks
        batch_size = embeddings.shape[0]
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        positive_mask.fill_diagonal_(0)  # Exclude self
        
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        pos_sim = sim_matrix[positive_mask > 0]
        neg_sim = sim_matrix[negative_mask > 0]
        
        if len(pos_sim) > 0 and len(neg_sim) > 0:
            loss = torch.mean(torch.clamp(self.margin - pos_sim, min=0)) + \
                   torch.mean(torch.clamp(neg_sim - (-self.margin), min=0))
            return loss
        else:
            return torch.tensor(0.0, device=embeddings.device)


class ClauseLevelAnalyzer:
    """
    Extract and analyze individual clauses in legal documents.
    
    Useful for:
    - Duplicate clause detection
    - Clause-level sentiment analysis
    - Comparative clause analysis
    """
    
    # Common clause markers in legal documents
    CLAUSE_MARKERS = [
        r'Article\s+\d+',
        r'Section\s+[\d\.]+',
        r'§\s*[\d\.]+',
        r'Clause\s+\d+',
        r'\d+\.\s+[A-Z]',  # Numbered clauses
    ]
    
    @staticmethod
    def extract_clauses(text: str) -> List[Dict[str, str]]:
        """
        Extract individual clauses from legal document.
        
        Returns:
            List of clauses with their identifiers and content
        """
        clauses = []
        current_clause = None
        
        for line in text.split('\n'):
            if re.match(r'^(\d+\.|Article|Section|Clause|§)', line):
                if current_clause:
                    clauses.append(current_clause)
                
                current_clause = {
                    'identifier': line.strip()[:50],  # First 50 chars as ID
                    'content': line.strip()
                }
            elif current_clause:
                current_clause['content'] += ' ' + line.strip()
        
        if current_clause:
            clauses.append(current_clause)
        
        return clauses
    
    @staticmethod
    def compute_clause_similarity(clause1: str, clause2: str) -> float:
        """
        Compute similarity between two clauses.
        
        Returns score between 0 (dissimilar) and 1 (identical)
        """
        # Normalize clauses
        clause1_words = set(clause1.lower().split())
        clause2_words = set(clause2.lower().split())
        
        intersection = len(clause1_words & clause2_words)
        union = len(clause1_words | clause2_words)
        
        return intersection / union if union > 0 else 0.0


class DocumentStructurePreserver(nn.Module):
    """
    Preserve document structure during processing.
    
    Important for legal documents which often have hierarchical structure:
    - Sections and subsections
    - Numbered lists
    - Clauses and sub-clauses
    """
    
    def __init__(self, max_document_sections: int = 50):
        super().__init__()
        self.max_sections = max_document_sections
        self.section_embeddings = nn.Embedding(max_document_sections, 768)
    
    @staticmethod
    def parse_document_structure(text: str) -> List[Dict]:
        """Parse hierarchical structure of document."""
        structure = []
        current_section = None
        
        for i, line in enumerate(text.split('\n')):
            # Detect section headers
            if re.match(r'^[A-Z][\w\s]+:$', line):
                current_section = {
                    'level': 1,
                    'title': line.strip(),
                    'content': [],
                    'line_numbers': [i]
                }
                structure.append(current_section)
            elif re.match(r'^\s{2}\d+\.\s', line):
                if current_section is None:
                    current_section = {'level': 0, 'title': 'Content', 'content': []}
                    structure.append(current_section)
                
                current_section['content'].append(line.strip())
                current_section['line_numbers'].append(i)
        
        return structure
    
    def forward(self, embeddings: torch.Tensor, section_indices: torch.Tensor) -> torch.Tensor:
        """
        Enhance embeddings with structural information.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_length, hidden_size]
            section_indices: Section index for each token [batch_size, seq_length]
        
        Returns:
            Enhanced embeddings with structural information
        """
        section_embs = self.section_embeddings(section_indices)
        return embeddings + 0.1 * section_embs  # Combine with structural info


class LegalDomainModule(nn.Module):
    """
    Complete domain-specific module combining all enhancements.
    
    Can be integrated into BERT for legal document processing.
    """
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.domain_attention = DomainSpecificAttention(hidden_size, num_heads)
        self.contrastive_loss = ContrastiveLearningLoss()
        self.structure_preserver = DocumentStructurePreserver()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                section_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process embeddings with domain-specific enhancements.
        """
        # Preserve document structure
        if section_indices is not None:
            embeddings = self.structure_preserver(embeddings, section_indices)
        
        # Apply domain-specific attention
        attn_out, attn_weights = self.domain_attention(embeddings, embeddings, embeddings)
        
        # Compute contrastive loss for better separation of document types
        contrastive_loss = self.contrastive_loss(embeddings[:, 0, :], labels)
        
        return {
            'embeddings': attn_out,
            'attention_weights': attn_weights,
            'contrastive_loss': contrastive_loss
        }


# Domain-specific configuration
LEGAL_DOMAIN_CONFIG = {
    'vocabulary_enhancement': {
        'enabled': True,
        'standardize_abbreviations': True,
        'expand_synonyms': True
    },
    'clause_analysis': {
        'enabled': True,
        'extract_clauses': True,
        'compute_similarities': True
    },
    'structure_preservation': {
        'enabled': True,
        'max_sections': 50
    },
    'contrastive_learning': {
        'enabled': True,
        'temperature': 0.07,
        'margin': 0.3
    },
    'domain_attention': {
        'enabled': True,
        'mask_citations': True,
        'mask_references': True
    }
}
