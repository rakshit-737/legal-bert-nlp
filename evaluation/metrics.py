"""
Evaluation metrics and visualization for legal BERT models
Computes accuracy, precision, recall, F1, and generates reports
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    auc, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EvaluationMetrics:
    """
    Comprehensive evaluation for classification models
    """
    
    def __init__(self, y_true: List[int], y_pred: List[int],
                 y_proba: Optional[np.ndarray] = None,
                 class_names: Optional[List[str]] = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.class_names = class_names or list(config.LABEL_TO_CLASS.values())
        self.num_classes = len(self.class_names)
    
    def get_metrics(self) -> Dict:
        """Calculate all metrics"""
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision_weighted": precision_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                self.y_true, self.y_pred, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                self.y_true, self.y_pred, average="macro", zero_division=0
            ),
            "f1_weighted": f1_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            ),
            "f1_macro": f1_score(
                self.y_true, self.y_pred, average="macro", zero_division=0
            ),
        }
        
        # Per-class metrics
        metrics["per_class"] = self._get_per_class_metrics()
        
        return metrics
    
    def _get_per_class_metrics(self) -> Dict:
        """Get metrics for each class"""
        per_class = {}
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        for class_name, class_idx in zip(self.class_names, range(self.num_classes)):
            per_class[class_name] = {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1": report[class_name]["f1-score"],
                "support": int(report[class_name]["support"])
            }
        
        return per_class
    
    def confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot and optionally save confusion matrix"""
        cm = self.confusion_matrix()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"}
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Confusion matrix saved to {save_path}")
        
        return plt
    
    def plot_roc_curves(self, save_path: Optional[str] = None):
        """Plot ROC curves for each class (one-vs-rest)"""
        if self.y_proba is None:
            print("⚠️  Probability predictions required for ROC curves")
            return None
        
        n_classes = self.num_classes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            # One-vs-rest encoding
            y_binary = np.array([1 if y == i else 0 for y in self.y_true])
            
            if len(np.unique(y_binary)) < 2:
                continue
            
            fpr, tpr, _ = roc_curve(y_binary, self.y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, label=f"{self.class_names[i]} (AUC={roc_auc:.3f})")
        
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right")
        ax.grid()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ ROC curves saved to {save_path}")
        
        return plt
    
    def plot_precision_recall(self, save_path: Optional[str] = None):
        """Plot precision-recall curves"""
        if self.y_proba is None:
            print("⚠️  Probability predictions required for P-R curves")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(self.num_classes):
            y_binary = np.array([1 if y == i else 0 for y in self.y_true])
            
            if len(np.unique(y_binary)) < 2:
                continue
            
            precision, recall, _ = precision_recall_curve(y_binary, self.y_proba[:, i])
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, lw=2, label=f"{self.class_names[i]} (AUC={pr_auc:.3f})")
        
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="best")
        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ P-R curves saved to {save_path}")
        
        return plt
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None):
        """Plot per-class metrics comparison"""
        metrics = self.get_metrics()
        per_class = metrics["per_class"]
        
        df = pd.DataFrame(per_class).T
        df = df[["precision", "recall", "f1"]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind="bar", ax=ax)
        ax.set_title("Per-Class Metrics")
        ax.set_ylabel("Score")
        ax.set_xlabel("Class")
        ax.legend(title="Metric")
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Metrics comparison saved to {save_path}")
        
        return plt
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        metrics = self.get_metrics()
        
        report = "=" * 60 + "\n"
        report += "📊 LEGAL BERT MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Overall metrics
        report += "📈 OVERALL METRICS\n"
        report += "-" * 60 + "\n"
        report += f"Accuracy:           {metrics['accuracy']:.4f}\n"
        report += f"Precision (macro):  {metrics['precision_macro']:.4f}\n"
        report += f"Precision (weighted): {metrics['precision_weighted']:.4f}\n"
        report += f"Recall (macro):     {metrics['recall_macro']:.4f}\n"
        report += f"Recall (weighted):  {metrics['recall_weighted']:.4f}\n"
        report += f"F1-Score (macro):   {metrics['f1_macro']:.4f}\n"
        report += f"F1-Score (weighted): {metrics['f1_weighted']:.4f}\n\n"
        
        # Per-class metrics
        report += "📋 PER-CLASS METRICS\n"
        report += "-" * 60 + "\n"
        per_class = metrics["per_class"]
        
        for class_name, scores in per_class.items():
            report += f"\n{class_name}:\n"
            report += f"  Precision: {scores['precision']:.4f}\n"
            report += f"  Recall:    {scores['recall']:.4f}\n"
            report += f"  F1-Score:  {scores['f1']:.4f}\n"
            report += f"  Support:   {scores['support']}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report)
            print(f"✅ Report saved to {save_path}")
        
        return report


class NERMetrics:
    """
    Evaluation metrics for Named Entity Recognition
    Uses token-level F1 score
    """
    
    def __init__(self, y_true: List[str], y_pred: List[str]):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def get_metrics(self) -> Dict:
        """Calculate NER metrics"""
        # Extract entities (simplistic approach)
        true_entities = self._extract_entities(self.y_true)
        pred_entities = self._extract_entities(self.y_pred)
        
        tp = len(true_entities & pred_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    @staticmethod
    def _extract_entities(tags: List[str]) -> set:
        """Extract entity spans from BIO tags"""
        entities = set()
        current_entity = None
        start_idx = 0
        
        for idx, tag in enumerate(tags):
            if tag.startswith("B-"):
                if current_entity:
                    entities.add((current_entity, start_idx, idx))
                current_entity = tag[2:]
                start_idx = idx
            elif tag.startswith("I-"):
                if current_entity != tag[2:]:
                    if current_entity:
                        entities.add((current_entity, start_idx, idx))
                    current_entity = tag[2:]
                    start_idx = idx
            else:  # "O" tag
                if current_entity:
                    entities.add((current_entity, start_idx, idx))
                    current_entity = None
        
        if current_entity:
            entities.add((current_entity, start_idx, len(tags)))
        
        return entities
