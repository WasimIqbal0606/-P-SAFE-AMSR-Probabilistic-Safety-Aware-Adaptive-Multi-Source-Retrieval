"""Generate paper tables from validated artifacts with fail-closed semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from paper.tools.build_evidence_tables import main as build_primary_tables


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "paper" / "tables"
CALIBRATION = ROOT / "results" / "calibration" / "calibration_metrics.json"
DATASETS = ["scifact", "fiqa", "nfcorpus", "arguana"]
LABELS = {
    "scifact": "SciFact",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
}


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Return a required key; never convert absent scientific evidence to zero."""
    if key not in mapping:
        raise KeyError(f"{context}: missing required key {key!r}")
    return mapping[key]


def require_path(mapping: dict[str, Any], keys: Iterable[str], context: str) -> Any:
    value: Any = mapping
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"{context}: missing required path {'.'.join(walked)!r}")
        value = value[key]
    return value


def evidence_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_calibration_table() -> None:
    data = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    rows: list[str] = []
    fields = [
        "n_positive",
        "n_samples",
        "positive_rate",
        "brier_score",
        "ece",
        "adaptive_ece",
        "auroc",
        "auprc",
        "calibration_slope",
        "calibration_intercept",
    ]
    for dataset in DATASETS:
        entry = require_path(data, [dataset, "42", "balanced"], str(CALIBRATION))
        for index, target in enumerate(("P_gain", "P_harm")):
            metrics = require(entry, target, f"{dataset}/42/balanced")
            values = {field: require(metrics, field, f"{dataset}/42/balanced/{target}") for field in fields}
            dataset_cell = LABELS[dataset] if index == 0 else ""
            target_cell = "$P_{\\mathrm{gain}}$" if target == "P_gain" else "$P_{\\mathrm{harm}}$"
            rows.append(
                f"{dataset_cell} & {target_cell} & {values['n_positive']}/{values['n_samples']} & "
                f"{values['positive_rate']:.3f} & {values['brier_score']:.4f} & "
                f"{values['ece']:.4f} & {values['adaptive_ece']:.4f} & "
                f"{values['auroc']:.3f} & {values['auprc']:.3f} & "
                f"{values['calibration_slope']:.2f} & {values['calibration_intercept']:+.2f} \\\\"
            )
    source_hash = evidence_hash([CALIBRATION])
    text = "\n".join(
        [
            f"% Source: {CALIBRATION.relative_to(ROOT).as_posix()}",
            f"% Evidence SHA256: {source_hash}",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Held-out calibration diagnostics for seed-42 Balanced routing. The probability models use internal cross-validated sigmoid calibration on the training split; routing thresholds are selected on validation.}",
            "\\label{tab:calibration}",
            "\\scriptsize",
            "\\begin{tabular}{llrrrrrrrrr}",
            "\\toprule",
            "Dataset & Target & Pos/Total & Prev. & Brier & ECE & Ad-ECE & AUROC & AUPRC & Slope & Intercept \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    (TABLES / "calibration_table.tex").write_text(text, encoding="utf-8")


def stamp_primary_tables() -> None:
    sources = list((ROOT / "results" / "validated").glob("**/extended_metrics.json"))
    sources += list((ROOT / "results" / "validated").glob("**/statistical_tests.json"))
    sources.append(ROOT / "configs" / "paper_experiment.yaml")
    source_hash = evidence_hash(sources)
    marker = "% Evidence SHA256:"
    for table in TABLES.glob("*.tex"):
        content = table.read_text(encoding="utf-8")
        if marker in content[:300]:
            continue
        table.write_text(
            f"% Sources: results/validated and configs/paper_experiment.yaml\n"
            f"% Evidence SHA256: {source_hash}\n{content}",
            encoding="utf-8",
        )


def generate_all_paper_tables() -> None:
    build_primary_tables()
    generate_calibration_table()
    stamp_primary_tables()
    print("TABLE GENERATION: PASS")


if __name__ == "__main__":
    generate_all_paper_tables()
