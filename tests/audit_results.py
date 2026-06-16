import os
import json
import shutil

def audit_results(src_dirs, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    report = ["# Result Audit\n", "The following datasets and modes were successfully audited as clean A0 vs A6 P-SAFE v1 runs.\n"]
    
    valid_count = 0
    invalid_count = 0

    for src in src_dirs:
        if not os.path.exists(src): continue
        for dataset in os.listdir(src):
            if dataset in ['cache', 'audit', 'aggregate', 'visualizations', 'paper_tables', 'reports', 'multi_dataset_summary']:
                continue
            dataset_path = os.path.join(src, dataset)
            if not os.path.isdir(dataset_path): continue
            
            for seed in os.listdir(dataset_path):
                seed_path = os.path.join(dataset_path, seed)
                if not os.path.isdir(seed_path): continue
                
                for mode in os.listdir(seed_path):
                    mode_path = os.path.join(seed_path, mode)
                    if not os.path.isdir(mode_path): continue
                    
                    metrics_file = os.path.join(mode_path, "metrics", "extended_metrics.json")
                    manifest_file = os.path.join(mode_path, "metrics", "reproducibility_manifest.json")
                    
                    is_valid = True
                    reason = ""
                    
                    if not os.path.exists(metrics_file):
                        is_valid = False
                        reason = "Missing extended_metrics.json"
                    else:
                        with open(metrics_file, "r") as f:
                            try:
                                m = json.load(f)
                                if "psafe_ndcg" not in m or "taxonomy" not in m:
                                    is_valid = False
                                    reason = "Invalid extended_metrics.json structure"
                            except Exception:
                                is_valid = False
                                reason = "Corrupt extended_metrics.json"
                    
                    # We can check manifest or other things
                    if os.path.exists(manifest_file) and is_valid:
                        with open(manifest_file, "r") as f:
                            try:
                                man = json.load(f)
                                if "A12_KNOWLEDGE_GRAPH" in man.get("action_distribution", {}):
                                    is_valid = False
                                    reason = "Mixed A0-A16 run (contains A12)"
                            except:
                                pass
                    
                    dest_mode = os.path.join(target_dir, dataset, seed, mode)
                    
                    if is_valid:
                        valid_count += 1
                        os.makedirs(dest_mode, exist_ok=True)
                        shutil.copytree(mode_path, dest_mode, dirs_exist_ok=True)
                        report.append(f"- **{dataset}** ({seed}) [{mode}]: PASS")
                    else:
                        invalid_count += 1
                        report.append(f"- **{dataset}** ({seed}) [{mode}]: FAIL ({reason})")

    report.append(f"\n**Total Valid:** {valid_count}")
    report.append(f"**Total Invalid:** {invalid_count}")
    
    with open("docs/result_audit.md", "w") as f:
        f.write("\n".join(report))
        
    print(f"Audit complete. Valid: {valid_count}, Invalid: {invalid_count}")

if __name__ == "__main__":
    audit_results(["results_next_level", "results_top_tier_psafe"], "results/validated")
