import os
import json
from ahrc.report_generator import generate_final_report

results_dir = 'results_psafe_amsr'
datasets = ['scifact', 'fiqa', 'nfcorpus']

for ds in datasets:
    ds_dir = os.path.join(results_dir, ds, 'Balanced')
    m_dir = os.path.join(ds_dir, 'metrics')
    if not os.path.exists(m_dir):
        print(f"Skipping {ds}, no metrics found.")
        continue

    with open(os.path.join(m_dir, 'aggregate_metrics.json')) as f: agg = json.load(f)
    with open(os.path.join(m_dir, 'safety_metrics.json')) as f: saf = json.load(f)
    with open(os.path.join(m_dir, 'statistical_tests.json')) as f: stat = json.load(f)
    
    cfg = {'num_queries': agg['Dense']['num_queries'], 'model': 'all-MiniLM-L6-v2'}
    
    # Try to load stats if available
    router_stats = {}
    try:
        with open(os.path.join(m_dir, 'action_distribution.json')) as f: dist = json.load(f)
        router_stats = {"P-SAFE": {"action_distribution": dist}}
    except:
        pass

    try:
        with open(os.path.join(m_dir, 'graph_contribution.json')) as f: graph_ablation = json.load(f)
    except:
        graph_ablation = None
        
    try:
        with open(os.path.join(m_dir, 'probability_calibration.json')) as f: prob_cal = json.load(f)
    except:
        prob_cal = None

    generate_final_report(
        dataset_name=f"BEIR/{ds}",
        results=agg,
        safety_metrics=saf,
        stat_reports=stat,
        config=cfg,
        router_stats=router_stats,
        graph_ablation=graph_ablation,
        prob_cal=prob_cal,
        output_path=os.path.join(ds_dir, 'final_report.md')
    )
    print(f"Regenerated {ds} final report.")

from ahrc.multi_dataset_runner import generate_multi_dataset_summary
generate_multi_dataset_summary('results_psafe_amsr', datasets)
print("Regenerated multi-dataset summary.")
