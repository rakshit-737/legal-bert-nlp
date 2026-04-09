"""
Advanced Optimization Techniques for BERT in Legal Document Processing

This module implements research-backed optimization strategies beyond standard fine-tuning:
- Learning rate scheduling strategies
- Mixed precision training
- Layer-wise learning rate decay
- Focal loss for imbalanced data
- Ensemble methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from typing import List, Tuple, Optional
import numpy as np


class LinearWarmupCosineAnnealingLR:
    """
    Combined linear warmup + cosine annealing learning rate scheduler.
    Research-backed approach for better convergence in BERT fine-tuning.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup phase
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lr):
            param_group['lr'] = base_lr * lr_scale
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class LayerWiseDecayScheduler:
    """
    Layer-wise learning rate decay for transformers.
    Earlier layers learn slower, later layers learn faster.
    Principle: Earlier layers contain general linguistic knowledge, 
    later layers contain task-specific knowledge.
    """
    def __init__(self, optimizer, base_lr: float, num_layers: int, decay_rate: float = 0.9):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.num_layers = num_layers
        self.decay_rate = decay_rate
        
        # Assign layer-wise learning rates
        self._assign_layer_lrs()
    
    def _assign_layer_lrs(self):
        """Assign decreasing learning rates to earlier layers."""
        for param_group in self.optimizer.param_groups:
            # Extract layer number from param group (if available)
            if 'layer_id' in param_group:
                layer_id = param_group['layer_id']
                param_group['lr'] = self.base_lr * (self.decay_rate ** (self.num_layers - layer_id))
    
    def get_layer_lrs(self):
        """Get current layer-wise learning rates."""
        return {f"layer_{i}": self.base_lr * (self.decay_rate ** (self.num_layers - i)) 
                for i in range(self.num_layers)}


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in document classification.
    
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Applied to NLP classification tasks.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing regularization to prevent overconfident predictions.
    
    Particularly useful for legal document classification where model uncertainty
    is important for calibration.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class EnsembleClassifier(nn.Module):
    """
    Ensemble of BERT-based classifiers for improved robustness.
    
    Combines predictions from multiple models using:
    - Weighted averaging
    - Voting mechanisms
    - Confidence-based aggregation
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction combining multiple models.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        
        Returns:
            Ensemble logits and confidence scores
        """
        logits_list = []
        
        for model in self.models:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_list.append(logits)
        
        # Stack and weight ensemble predictions
        logits_stack = torch.stack(logits_list)  # [num_models, batch_size, num_classes]
        weights = self.weights.to(logits_stack.device).view(-1, 1, 1)
        
        ensemble_logits = (logits_stack * weights).sum(dim=0)  # [batch_size, num_classes]
        
        # Confidence = max probability
        ensemble_probs = F.softmax(ensemble_logits, dim=-1)
        confidence = torch.max(ensemble_probs, dim=-1)[0]
        
        return ensemble_logits, confidence


class MixedPrecisionTrainer:
    """
    Mixed precision training (FP16/FP32) for faster training and reduced memory.
    
    Research shows minimal accuracy loss with significant speedup (1.5-3x).
    """
    def __init__(self, model: nn.Module, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or torch.cuda.amp.GradScaler()
    
    def training_step(self, batch, criterion):
        """
        Single training step with mixed precision.
        """
        input_ids = batch['input_ids'].to(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        labels = batch['labels'].to(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        
        with torch.cuda.amp.autocast():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()


class TextAugmentationStrategy:
    """
    Data augmentation strategies for legal text to prevent overfitting.
    
    Techniques:
    - Token shuffling (limited)
    - Synonym replacement
    - EDA (Easy Data Augmentation)
    """
    def __init__(self, num_augmentations: int = 4):
        self.num_augmentations = num_augmentations
    
    def random_insertion(self, text: str, synonyms_dict: dict, num_words: int = 2) -> str:
        """Insert random synonyms into text."""
        words = text.split()
        for _ in range(num_words):
            if words:
                idx = np.random.randint(0, len(words))
                word = words[idx]
                if word in synonyms_dict:
                    synonym = np.random.choice(synonyms_dict[word])
                    words.insert(idx, synonym)
        return ' '.join(words)
    
    def random_swap(self, text: str, num_swaps: int = 2) -> str:
        """Randomly swap words in text."""
        words = text.split()
        for _ in range(num_swaps):
            if len(words) > 1:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def random_deletion(self, text: str, probability: float = 0.1) -> str:
        """Randomly delete words with given probability."""
        words = text.split()
        new_words = [w for w in words if np.random.random() > probability]
        return ' '.join(new_words) if new_words else text
    
    def augment_batch(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Generate augmented samples for entire batch."""
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        for text, label in zip(texts, labels):
            for _ in range(self.num_augmentations):
                aug_type = np.random.choice(['swap', 'deletion', 'insertion'])
                
                if aug_type == 'swap':
                    augmented_text = self.random_swap(text)
                elif aug_type == 'deletion':
                    augmented_text = self.random_deletion(text)
                else:
                    augmented_text = self.random_insertion(text, {}, num_words=1)
                
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
        
        return augmented_texts, augmented_labels


class GradientAccumulation:
    """
    Gradient accumulation for effective batch size increase with memory constraints.
    
    Useful for legal document processing where documents/batches can be large.
    """
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
    
    def should_update(self) -> bool:
        """Check if optimizer should be stepped."""
        return (self.step_counter + 1) % self.accumulation_steps == 0
    
    def backward_pass(self, loss: torch.Tensor):
        """Backward pass with scaling by accumulation steps."""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self.step_counter += 1
    
    def optimizer_step(self, optimizer, scaler=None):
        """Execute optimizer step with optional gradient clipping."""
        if scaler:
            scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
        
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        self.step_counter = 0


# Example usage configuration
OPTIMIZATION_CONFIG = {
    'learning_rate_schedule': 'linear_warmup_cosine',
    'warmup_fraction': 0.1,
    'layer_wise_decay': {
        'enabled': True,
        'decay_rate': 0.9,
        'num_layers': 12
    },
    'loss_function': 'focal_loss',  # or 'label_smoothing'
    'focal_loss_params': {
        'alpha': 0.25,
        'gamma': 2.0
    },
    'label_smoothing': {
        'enabled': True,
        'smoothing': 0.1
    },
    'gradient_accumulation': {
        'enabled': True,
        'steps': 4
    },
    'mixed_precision': {
        'enabled': True
    },
    'data_augmentation': {
        'enabled': True,
        'num_augmentations': 4
    },
    'ensemble': {
        'enabled': False,
        'num_models': 3
    }
}
