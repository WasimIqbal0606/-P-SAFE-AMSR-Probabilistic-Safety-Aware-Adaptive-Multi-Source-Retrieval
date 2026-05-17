"""
Safe-AMSR-SE v4 — CLI Entry Point

Usage:
    .\.venv\Scripts\python.exe run_safe_amsr_v4.py
    .\.venv\Scripts\python.exe run_safe_amsr_v4.py --dataset fiqa
    .\.venv\Scripts\python.exe run_safe_amsr_v4.py --multi --datasets scifact fiqa nfcorpus
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser(description="Safe-AMSR-SE v4")
    p.add_argument("--source", default="beir")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--multi", action="store_true")
    p.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "nfcorpus"])
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--results-dir", default="results_safe_amsr_v4")
    args = p.parse_args()

    if args.multi:
        from ahrc.multi_dataset_runner import run_multi_dataset
        run_multi_dataset(datasets=args.datasets, results_dir=args.results_dir,
                          model_name=args.model, device=args.device,
                          max_docs=args.max_docs, max_queries=args.max_queries)
    else:
        from ahrc.safe_experiment_runner import run_safe_experiment
        kwargs = {"dataset_name": args.dataset}
        if args.max_docs: kwargs["max_docs"] = args.max_docs
        if args.max_queries: kwargs["max_queries"] = args.max_queries
        run_safe_experiment(source=args.source, results_dir=args.results_dir,
                            model_name=args.model, device=args.device, **kwargs)

if __name__ == "__main__":
    main()
