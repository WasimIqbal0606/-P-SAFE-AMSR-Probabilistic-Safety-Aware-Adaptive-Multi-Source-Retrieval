"""Guard against reconstructing publication splits from test-side evidence.

Historical primary runs did not commit their train and validation query IDs or
their original feature matrices. Those objects cannot be reconstructed from the
test CSVs without contaminating the scientific record. A fresh primary run must
write ``split_manifest.json`` directly from the live split indices.
"""


def main() -> None:
    raise RuntimeError(
        "Split features cannot be reconstructed from validated test artifacts. "
        "Rerun the canonical primary experiment to create real train/validation/"
        "test feature matrices and split_manifest.json files."
    )


if __name__ == "__main__":
    main()
