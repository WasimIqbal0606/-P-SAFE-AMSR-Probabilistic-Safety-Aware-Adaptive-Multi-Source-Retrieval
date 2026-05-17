"""
Safe-AMSR-SE v4 — Leakage-Safe Query Splits
Ensures router trains on TRAIN, tunes on VAL, reports on TEST.
No ground-truth information leaks into the test evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class QuerySplit:
    """Holds indices for train/val/test splits."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    seed: int = 42

    @property
    def n_train(self) -> int: return len(self.train_idx)
    @property
    def n_val(self) -> int: return len(self.val_idx)
    @property
    def n_test(self) -> int: return len(self.test_idx)

    def summary(self) -> Dict:
        return {
            "n_train": self.n_train, "n_val": self.n_val, "n_test": self.n_test,
            "train_pct": self.n_train / (self.n_train + self.n_val + self.n_test),
            "val_pct": self.n_val / (self.n_train + self.n_val + self.n_test),
            "test_pct": self.n_test / (self.n_train + self.n_val + self.n_test),
            "seed": self.seed,
        }


def create_stratified_split(
    dense_ndcg: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    easy_threshold: float = 0.5,
    seed: int = 42,
) -> QuerySplit:
    """
    Create stratified train/val/test split.
    Stratifies by difficulty (easy vs hard) so each split has similar proportions.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    n = len(dense_ndcg)

    easy_idx = np.where(dense_ndcg > easy_threshold)[0]
    hard_idx = np.where(dense_ndcg <= easy_threshold)[0]

    def _split_group(indices):
        perm = rng.permutation(indices)
        n_g = len(perm)
        n_train = max(1, int(n_g * train_ratio))
        n_val = max(1, int(n_g * val_ratio))
        return (perm[:n_train],
                perm[n_train:n_train + n_val],
                perm[n_train + n_val:])

    e_tr, e_va, e_te = _split_group(easy_idx)
    h_tr, h_va, h_te = _split_group(hard_idx)

    split = QuerySplit(
        train_idx=np.sort(np.concatenate([e_tr, h_tr])),
        val_idx=np.sort(np.concatenate([e_va, h_va])),
        test_idx=np.sort(np.concatenate([e_te, h_te])),
        seed=seed,
    )

    print(f"   Split: train={split.n_train}, val={split.n_val}, test={split.n_test}")
    print(f"   Easy: train={len(e_tr)}, val={len(e_va)}, test={len(e_te)}")
    print(f"   Hard: train={len(h_tr)}, val={len(h_va)}, test={len(h_te)}")
    return split


def create_kfold_splits(
    dense_ndcg: np.ndarray,
    n_folds: int = 5,
    easy_threshold: float = 0.5,
    seed: int = 42,
) -> List[QuerySplit]:
    """K-Fold cross-validation splits with stratification."""
    rng = np.random.default_rng(seed)
    n = len(dense_ndcg)
    easy_idx = np.where(dense_ndcg > easy_threshold)[0]
    hard_idx = np.where(dense_ndcg <= easy_threshold)[0]

    def _kfold(indices):
        perm = rng.permutation(indices)
        folds = np.array_split(perm, n_folds)
        return folds

    easy_folds = _kfold(easy_idx)
    hard_folds = _kfold(hard_idx)
    splits = []

    for k in range(n_folds):
        test_idx = np.sort(np.concatenate([easy_folds[k], hard_folds[k]]))
        val_k = (k + 1) % n_folds
        val_idx = np.sort(np.concatenate([easy_folds[val_k], hard_folds[val_k]]))
        train_folds = [i for i in range(n_folds) if i != k and i != val_k]
        train_idx = np.sort(np.concatenate(
            [easy_folds[i] for i in train_folds] +
            [hard_folds[i] for i in train_folds]
        ))
        splits.append(QuerySplit(train_idx=train_idx, val_idx=val_idx,
                                  test_idx=test_idx, seed=seed))

    return splits
