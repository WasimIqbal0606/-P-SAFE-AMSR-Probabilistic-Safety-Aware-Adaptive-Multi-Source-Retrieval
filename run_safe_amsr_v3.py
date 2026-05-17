"""
Safe-AMSR-SE v3 — CLI Entry Point

Usage:
    # Single dataset (SciFact)
    python run_safe_amsr_v3.py

    # Specific dataset
    python run_safe_amsr_v3.py --dataset fiqa

    # Multi-dataset
    python run_safe_amsr_v3.py --multi --datasets scifact fiqa nfcorpus

    # With GPU
    python run_safe_amsr_v3.py --device auto
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Safe-AMSR-SE v3: Adaptive Retrieval Router for IR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_safe_amsr_v3.py                           # SciFact only
  python run_safe_amsr_v3.py --dataset fiqa            # FiQA only
  python run_safe_amsr_v3.py --multi                   # All mandatory datasets
  python run_safe_amsr_v3.py --device auto              # Auto GPU detection
  python run_safe_amsr_v3.py --max-docs 5000            # Limit corpus size
        """
    )

    parser.add_argument("--source", default="beir",
                        choices=["synthetic", "beir", "msmarco", "trec-dl"])
    parser.add_argument("--dataset", default="scifact",
                        help="BEIR dataset name (scifact, fiqa, nfcorpus, trec-covid)")
    parser.add_argument("--multi", action="store_true",
                        help="Run on multiple datasets")
    parser.add_argument("--datasets", nargs="+",
                        default=["scifact", "fiqa", "nfcorpus"],
                        help="Datasets for multi-dataset mode")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-Transformer model name")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for cross-encoder")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit corpus size")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries")
    parser.add_argument("--results-dir", default="results_safe_amsr_v3",
                        help="Output directory")

    args = parser.parse_args()

    if args.multi:
        from ahrc.multi_dataset_runner import run_multi_dataset
        run_multi_dataset(
            datasets=args.datasets,
            results_dir=args.results_dir,
            model_name=args.model,
            device=args.device,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )
    else:
        from ahrc.safe_experiment_runner import run_safe_experiment
        kwargs = {"dataset_name": args.dataset} if args.source == "beir" else {}
        if args.max_docs:
            kwargs["max_docs"] = args.max_docs
        if args.max_queries:
            kwargs["max_queries"] = args.max_queries

        run_safe_experiment(
            source=args.source,
            results_dir=args.results_dir,
            model_name=args.model,
            device=args.device,
            **kwargs,
        )


if __name__ == "__main__":
    main()
