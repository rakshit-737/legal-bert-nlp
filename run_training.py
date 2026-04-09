#!/usr/bin/env python
"""
run_training.py  —  One-click training entry-point
===================================================

Trains OptiBERT on legal documents (dataset auto-selected),
saves checkpoint + evaluation artefacts, then prints a
comparison table matching the paper's Table 2 / Table 3.

Usage:
    python run_training.py                        # defaults
    python run_training.py --source synthetic     # guaranteed offline
    python run_training.py --epochs 3 --batch 4  # fast smoke-test
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   OptiBERT — Legal Document Processing                      ║
║   Based on: Optimization of BERT Algorithms for Deep        ║
║   Contextual Analysis and Automation in Legal Document       ║
║   Processing (ICCCNT 2024, IIT Mandi)                       ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_comparison(test_metrics: dict):
    """Print the model-comparison table from the paper."""
    baselines = config.BASELINE_RESULTS
    print("\n" + "═" * 68)
    print("  MODEL COMPARISON TABLE  (paper Table 2 / Table 3 equivalent)")
    print("═" * 68)
    header = f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("─" * 68)

    for model_name, m in baselines.items():
        tag = " ← paper" if model_name == "OptiBERT" else ""
        print(
            f"{model_name:<14} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f} {m['f1']:>10.4f}{tag}"
        )

    # Our result
    tag = " ← this run"
    print("─" * 68)
    print(
        f"{'Our OptiBERT':<14} {test_metrics['accuracy']:>10.4f}"
        f" {test_metrics['precision']:>10.4f}"
        f" {test_metrics['recall']:>10.4f}"
        f" {test_metrics['f1']:>10.4f}{tag}"
    )
    print("═" * 68)


def check_deps():
    missing = []
    for pkg in ["torch", "transformers", "sklearn", "numpy", "datasets"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train OptiBERT Legal Classifier")
    parser.add_argument("--source",  default="auto",
                        choices=["auto", "synthetic", "huggingface"],
                        help="Dataset source")
    parser.add_argument("--epochs",  type=int,   default=None, help="Number of epochs")
    parser.add_argument("--lr",      type=float, default=None, help="Learning rate")
    parser.add_argument("--batch",   type=int,   default=None, help="Batch size")
    parser.add_argument("--model",   default=None,
                        help=f"HF model name (default: {config.DEFAULT_MODEL})")
    args = parser.parse_args()

    print_banner()
    check_deps()

    from training.train_classifier import TrainingPipeline

    pipeline = TrainingPipeline(model_name=args.model)
    test_metrics = pipeline.run(
        source=args.source,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
    )

    print_comparison(test_metrics)

    # Save combined report
    report_path = Path(config.RESULTS_DIR) / "final_report.json"
    full_report = {
        "model":        args.model or config.DEFAULT_MODEL,
        "source":       args.source,
        "test_metrics": test_metrics,
        "history":      pipeline.history,
        "baselines":    config.BASELINE_RESULTS,
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n📄 Full report saved → {report_path}")
    print("\n✅  Training complete. Run the Streamlit app:")
    print("    streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
