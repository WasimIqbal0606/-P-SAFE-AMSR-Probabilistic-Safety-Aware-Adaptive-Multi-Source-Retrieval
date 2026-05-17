"""
P-SAFE-AMSR — CLI Entry Point

Usage:
    .\.venv\Scripts\python.exe run_psafe_amsr.py --dataset scifact
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser(description="P-SAFE-AMSR")
    p.add_argument("--source", default="beir")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--results-dir", default="results_psafe_amsr")
    args = p.parse_args()

    from ahrc.psafe_experiment_runner import run_psafe_experiment
    kwargs = {"dataset_name": args.dataset}
    if args.max_docs: kwargs["max_docs"] = args.max_docs
    if args.max_queries: kwargs["max_queries"] = args.max_queries
    
    run_psafe_experiment(source=args.source, results_dir=args.results_dir,
                         model_name=args.model, device=args.device, **kwargs)

if __name__ == "__main__":
    main()
