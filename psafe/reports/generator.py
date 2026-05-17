import os

def generate_limitations(model_name: str) -> str:
    """Generate dynamic limitations text based on the model."""
    text = "## Limitations\n\n"
    text += "- Current graph is synthetic kNN based on dense embeddings and does not independently drive gains.\n"
    
    if "bge-m3" in model_name.lower():
        text += "- Further experiments should compare against E5-large, Qwen3 embeddings, SPLADE, ColBERT, and stronger reranker variants.\n"
    else:
        text += "- Results need stronger baselines such as SPLADE, ColBERT, BGE-M3, and E5/BGE dense models.\n"
        
    text += "- Some datasets show protection rather than absolute nDCG improvement.\n"
    text += "- More datasets and larger test splits are required before strong journal claims.\n"
    return text

def generate_behavior_claim(is_multi_dataset: bool, dataset_name: str, behavior_type: str) -> str:
    """Generate behavior claim avoiding overclaiming on single datasets."""
    if is_multi_dataset:
        return "B-P-SAFE demonstrates consistent and adaptive behavior across different dataset behaviours."
    else:
        return f"On this dataset, B-P-SAFE demonstrates {behavior_type} behaviour."

def write_reproducibility_manifest(out_dir: str, forced_hybrid: bool, skipped_baselines: list, gpu_fallback: list):
    """Write reproducibility manifest."""
    import json
    manifest_path = os.path.join(out_dir, "reproducibility_manifest.json")
    
    warning = ""
    if forced_hybrid:
        warning = "WARNING: min_hybrid_rate was explicitly forced > 0.0 in final test. This violates strict evaluation protocol."
    else:
        warning = "min_hybrid_rate was safely forced to 0.0 in final test."

    data = {
        "forced_hybrid_warning": warning,
        "skipped_baselines": skipped_baselines,
        "gpu_fallback_behavior": gpu_fallback
    }
    
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=4)
