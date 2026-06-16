"""
AMSR-SE — Dataset Interface
Unified interface for loading retrieval benchmarks from multiple sources.

Supports:
  1. Synthetic benchmark (existing AHRC dataset)
  2. BEIR datasets (scifact, fiqa, nfcorpus, trec-covid, nq, etc.)
  3. MS MARCO passage dev subset
  4. TREC DL 2019/2020 (hard-query subsets)

Interface:
  load_corpus()  → Dict[str, Dict]  {doc_id: {"text": ..., "title": ...}}
  load_queries()  → Dict[str, str]   {query_id: text}
  load_qrels()    → Dict[str, Dict]  {query_id: {doc_id: relevance}}
  evaluate()      → runs full AMSR-SE pipeline on loaded benchmark
"""

import os
import json
import time
import shutil
import zipfile
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# Abstract base class
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkData:
    """Loaded benchmark data, ready for retrieval experiments."""
    name: str
    corpus: Dict[str, Dict]       # {doc_id: {"text": ..., "title": ...}}
    queries: Dict[str, str]       # {query_id: text}
    qrels: Dict[str, Dict[str, int]]  # {query_id: {doc_id: relevance}}
    corpus_texts: List[str]       # ordered list of doc texts (for embedding)
    corpus_ids: List[str]         # ordered list of doc ids
    query_texts: List[str]        # ordered list of query texts
    query_ids: List[str]          # ordered list of query ids

    @property
    def num_docs(self) -> int:
        return len(self.corpus)

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    def summary(self) -> str:
        avg_rels = 0
        if self.qrels:
            rel_counts = [sum(1 for v in rels.values() if v >= 1) for rels in self.qrels.values()]
            avg_rels = np.mean(rel_counts) if rel_counts else 0

        return (
            f"[Benchmark] {self.name}\n"
            f"   Corpus:  {self.num_docs:,} documents\n"
            f"   Queries: {self.num_queries:,}\n"
            f"   Avg relevant docs/query: {avg_rels:.1f}"
        )


class DatasetLoader(ABC):
    """Abstract loader interface."""

    @abstractmethod
    def load(self, **kwargs) -> BenchmarkData:
        """Load and return benchmark data."""
        pass

    @staticmethod
    def _build_ordered_lists(
        corpus: Dict[str, Dict], queries: Dict[str, str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Convert dicts to ordered lists for embedding."""
        corpus_ids = sorted(corpus.keys())
        corpus_texts = []
        for cid in corpus_ids:
            doc = corpus[cid]
            title = doc.get("title", "")
            text = doc.get("text", "")
            corpus_texts.append(f"{title} {text}".strip() if title else text)

        query_ids = sorted(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        return corpus_texts, corpus_ids, query_texts, query_ids


# ═══════════════════════════════════════════════════════════════════════
# 1. Synthetic Benchmark (existing AHRC)
# ═══════════════════════════════════════════════════════════════════════

class SyntheticLoader(DatasetLoader):
    """Load the existing AHRC synthetic benchmark."""

    def load(self, data_dir: str = "ahrc_data", config=None, **kwargs) -> BenchmarkData:
        """Load from saved AHRC data directory, or generate if missing."""
        from .config import AHRCConfig
        from .dataset_generator import DatasetGenerator, build_dataset

        if config is None:
            config = AHRCConfig()

        if os.path.exists(os.path.join(data_dir, "tasks.json")):
            dataset = DatasetGenerator.load(data_dir, config)
        else:
            dataset = build_dataset(config, data_dir)

        # Convert to BenchmarkData format
        corpus = {}
        for task in dataset.tasks:
            corpus[task.id] = {"text": task.description, "title": ""}

        queries = {}
        for q in dataset.queries:
            queries[q.id] = q.text

        qrels = {}
        for q in dataset.queries:
            qrels[q.id] = q.relevance

        corpus_texts, corpus_ids, query_texts, query_ids = (
            self._build_ordered_lists(corpus, queries)
        )

        return BenchmarkData(
            name=f"Synthetic ({len(corpus):,} tasks)",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            query_texts=query_texts,
            query_ids=query_ids,
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. BEIR Datasets
# ═══════════════════════════════════════════════════════════════════════

BEIR_DATASETS = {
    "scifact":     "Scientific fact verification (5.2K docs, 300 queries)",
    "fiqa":        "Financial opinion QA (57K docs, 648 queries)",
    "nfcorpus":    "Nutrition/health retrieval (3.6K docs, 323 queries)",
    "trec-covid":  "COVID-19 scientific search (171K docs, 50 queries)",
    "nq":          "Natural Questions (2.7M docs, 3.5K queries) — LARGE",
    "hotpotqa":    "Multi-hop QA (5.2M docs, 7.4K queries) — LARGE",
    "arguana":     "Argument retrieval (8.7K docs, 1.4K queries)",
    "webis-touche2020": "Argument search (382K docs, 49 queries)",
    "quora":       "Duplicate question detection (523K docs, 10K queries)",
    "dbpedia-entity": "Entity search (4.6M docs, 400 queries) — LARGE",
    "fever":       "Fact verification (5.4M docs, 6.7K queries) — LARGE",
    "climate-fever": "Climate fact check (5.4M docs, 1.5K queries) — LARGE",
    "scidocs":     "Scientific document retrieval (25K docs, 1K queries)",
}


class BEIRLoader(DatasetLoader):
    """
    Load any BEIR benchmark dataset.

    Uses the official beir library to download and parse datasets.
    Small datasets (scifact, fiqa, nfcorpus, arguana) are recommended
    for development. Large datasets (nq, hotpotqa) require significant
    disk space and RAM.
    """

    @staticmethod
    def _validate_zip(zip_path: str) -> None:
        if not os.path.exists(zip_path):
            return
        if not zipfile.is_zipfile(zip_path):
            raise zipfile.BadZipFile(f"{zip_path} exists but is not a valid zip file")
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_member = zf.testzip()
        if bad_member:
            raise zipfile.BadZipFile(
                f"{zip_path} failed zip integrity check at member {bad_member}"
            )

    @staticmethod
    def _validate_extracted(data_path: str, split: str) -> None:
        required = [
            os.path.join(data_path, "corpus.jsonl"),
            os.path.join(data_path, "queries.jsonl"),
            os.path.join(data_path, "qrels"),
            os.path.join(data_path, "qrels", f"{split}.tsv"),
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Extracted BEIR dataset is incomplete; missing: "
                + ", ".join(missing)
            )

    @staticmethod
    def _remove_dataset_cache(dataset_name: str, out_dir: str) -> None:
        allowed_root = os.path.abspath(out_dir)
        targets = [
            os.path.join(out_dir, f"{dataset_name}.zip"),
            os.path.join(out_dir, dataset_name),
        ]
        for target in targets:
            abs_target = os.path.abspath(target)
            if (
                abs_target != allowed_root
                and abs_target.startswith(allowed_root + os.sep)
                and os.path.exists(abs_target)
            ):
                if os.path.isdir(abs_target):
                    shutil.rmtree(abs_target)
                else:
                    os.remove(abs_target)

    def _download_validate_and_extract(
        self, util, url: str, out_dir: str, dataset_name: str, split: str
    ) -> str:
        zip_path = os.path.join(out_dir, f"{dataset_name}.zip")
        data_path = os.path.join(out_dir, dataset_name)

        if not os.path.exists(zip_path):
            print(f"   [Downloading] {dataset_name}...")
            util.download_url(url, zip_path)

        self._validate_zip(zip_path)

        if not os.path.isdir(data_path):
            print(f"   [Unzipping] {dataset_name}...")
            util.unzip(zip_path, out_dir)

        self._validate_extracted(data_path, split)
        return data_path

    def load(
        self,
        dataset_name: str = "scifact",
        split: str = "test",
        data_dir: str = "datasets",
        max_docs: Optional[int] = None,
        max_queries: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkData:
        """
        Download and load a BEIR dataset.

        Args:
            dataset_name: one of BEIR_DATASETS keys.
            split: 'test', 'dev', or 'train'.
            data_dir: where to cache downloaded files.
            max_docs: optional limit on corpus size (for memory).
            max_queries: optional limit on number of queries.
        """
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        print(f"\n[Loading] BEIR dataset: {dataset_name} (split={split})")

        # Download if needed
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        out_dir = os.path.join(data_dir, "beir")
        os.makedirs(out_dir, exist_ok=True)

        data_path = os.path.join(out_dir, dataset_name)
        zip_path = os.path.join(out_dir, f"{dataset_name}.zip")

        try:
            if os.path.isdir(data_path):
                self._validate_extracted(data_path, split)
                print(f"   [Cached] Using cached {dataset_name}")
            else:
                self._validate_zip(zip_path)
                data_path = self._download_validate_and_extract(
                    util, url, out_dir, dataset_name, split
                )
        except (zipfile.BadZipFile, FileNotFoundError) as cache_error:
            print(f"   [Cache invalid] {cache_error}")
            print(f"   [Cache cleanup] Removing only cached files for {dataset_name}")
            self._remove_dataset_cache(dataset_name, out_dir)
            try:
                data_path = self._download_validate_and_extract(
                    util, url, out_dir, dataset_name, split
                )
            except Exception as redownload_error:
                raise RuntimeError(
                    f"Failed to redownload BEIR/{dataset_name}: {redownload_error}"
                ) from redownload_error

        # Load via GenericDataLoader
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

        print(f"   [Loaded] {len(corpus):,} docs, {len(queries):,} queries, "
              f"{len(qrels):,} judged queries")

        # Optional subsample for memory
        if max_docs and len(corpus) > max_docs:
            print(f"   [Subsampling] corpus to {max_docs:,} docs")
            # Keep docs that appear in qrels first
            relevant_docs: Set[str] = set()
            for qid, doc_rels in qrels.items():
                for did in doc_rels:
                    relevant_docs.add(did)

            # Always keep relevant docs
            keep_ids = list(relevant_docs)
            remaining = [did for did in corpus if did not in relevant_docs]
            need = max_docs - len(keep_ids)
            if need > 0:
                import random
                random.seed(42)
                keep_ids.extend(random.sample(remaining, min(need, len(remaining))))

            corpus = {did: corpus[did] for did in keep_ids if did in corpus}

        if max_queries and len(queries) > max_queries:
            print(f"   [Subsampling] queries to {max_queries}")
            import random
            random.seed(42)
            keep_qids = random.sample(list(queries.keys()), max_queries)
            queries = {qid: queries[qid] for qid in keep_qids}
            qrels = {qid: qrels[qid] for qid in keep_qids if qid in qrels}

        # Convert qrels format: BEIR uses {qid: {did: int}}
        # Ensure all values are ints
        clean_qrels = {}
        for qid, doc_rels in qrels.items():
            clean_qrels[qid] = {did: int(rel) for did, rel in doc_rels.items()}

        corpus_texts, corpus_ids, query_texts, query_ids = (
            self._build_ordered_lists(corpus, queries)
        )

        return BenchmarkData(
            name=f"BEIR/{dataset_name}",
            corpus=corpus,
            queries=queries,
            qrels=clean_qrels,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            query_texts=query_texts,
            query_ids=query_ids,
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. MS MARCO Passage Dev
# ═══════════════════════════════════════════════════════════════════════

class MSMARCOLoader(DatasetLoader):
    """
    Load MS MARCO passage ranking dev set via ir_datasets.

    This is the standard IR evaluation benchmark with ~8.8M passages
    and ~6980 dev queries. We subsample for tractability.
    """

    def load(
        self,
        variant: str = "dev/small",
        max_docs: int = 50000,
        max_queries: int = 500,
        **kwargs,
    ) -> BenchmarkData:
        """
        Load MS MARCO passage ranking.

        Args:
            variant: 'dev/small' (6980 queries) or 'dev' (full).
            max_docs: limit corpus size for embedding tractability.
            max_queries: limit number of evaluation queries.
        """
        import ir_datasets

        dataset_id = f"msmarco-passage/{variant}"
        print(f"\n[Loading] MS MARCO: {dataset_id}")
        print(f"   [Downloading] This may download ~3GB on first run...")

        dataset = ir_datasets.load(dataset_id)

        # Load qrels first to identify relevant docs
        print("   [Loading] qrels...")
        qrels: Dict[str, Dict[str, int]] = {}
        relevant_docs: Set[str] = set()
        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            did = str(qrel.doc_id)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(qrel.relevance)
            if qrel.relevance >= 1:
                relevant_docs.add(did)

        # Load queries
        print("   [Loading] queries...")
        queries: Dict[str, str] = {}
        for query in dataset.queries_iter():
            qid = str(query.query_id)
            if qid in qrels:
                queries[qid] = query.text

        # Subsample queries
        if max_queries and len(queries) > max_queries:
            import random
            random.seed(42)
            keep_qids = random.sample(list(queries.keys()), max_queries)
            queries = {qid: queries[qid] for qid in keep_qids}
            qrels = {qid: qrels[qid] for qid in keep_qids if qid in qrels}
            # Recompute relevant docs
            relevant_docs = set()
            for qid, doc_rels in qrels.items():
                for did, rel in doc_rels.items():
                    if rel >= 1:
                        relevant_docs.add(did)

        # Load docs (prioritize relevant ones)
        print(f"   [Loading] passages (target: {max_docs:,})...")
        # Use the main passage dataset for docs
        try:
            main_dataset = ir_datasets.load("msmarco-passage")
            docstore = main_dataset.docs_store()

            corpus: Dict[str, Dict] = {}

            # Always include relevant docs
            for did in relevant_docs:
                try:
                    doc = docstore.get(did)
                    corpus[did] = {"text": doc.text, "title": ""}
                except Exception:
                    pass

            # Fill remaining with random docs
            remaining_budget = max_docs - len(corpus)
            if remaining_budget > 0:
                count = 0
                for doc in main_dataset.docs_iter():
                    did = str(doc.doc_id)
                    if did not in corpus:
                        corpus[did] = {"text": doc.text, "title": ""}
                        count += 1
                        if count >= remaining_budget:
                            break

        except Exception as e:
            print(f"   [Warning] Direct doc loading failed ({e}), trying iterator...")
            corpus = {}
            main_dataset = ir_datasets.load("msmarco-passage")
            count = 0
            for doc in main_dataset.docs_iter():
                did = str(doc.doc_id)
                corpus[did] = {"text": doc.text, "title": ""}
                count += 1
                if count >= max_docs:
                    break

        print(f"   [Loaded] {len(corpus):,} passages, {len(queries):,} queries, "
              f"{len(relevant_docs):,} relevant docs")

        corpus_texts, corpus_ids, query_texts, query_ids = (
            self._build_ordered_lists(corpus, queries)
        )

        return BenchmarkData(
            name=f"MS-MARCO/{variant}",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            query_texts=query_texts,
            query_ids=query_ids,
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. TREC DL (hard-query subset)
# ═══════════════════════════════════════════════════════════════════════

class TRECDLLoader(DatasetLoader):
    """
    Load TREC Deep Learning track queries.
    These are a curated set of "hard" queries with graded relevance,
    evaluated over the MS MARCO passage corpus.
    """

    def load(
        self,
        year: int = 2019,
        max_docs: int = 50000,
        **kwargs,
    ) -> BenchmarkData:
        """
        Load TREC DL 2019 or 2020.

        Args:
            year: 2019 or 2020.
            max_docs: limit corpus size.
        """
        import ir_datasets

        dataset_id = f"msmarco-passage/trec-dl-{year}"
        print(f"\n[Loading] TREC DL {year}: {dataset_id}")

        try:
            dataset = ir_datasets.load(dataset_id)
        except Exception as e:
            print(f"   [Error] Failed to load TREC DL {year}: {e}")
            print(f"   [Info] Try: pip install ir_datasets[trec-dl]")
            raise

        # Load qrels
        print("   [Loading] qrels...")
        qrels: Dict[str, Dict[str, int]] = {}
        relevant_docs: Set[str] = set()
        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            did = str(qrel.doc_id)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(qrel.relevance)
            if qrel.relevance >= 1:
                relevant_docs.add(did)

        # Load queries
        print("   [Loading] queries...")
        queries = {}
        for query in dataset.queries_iter():
            qid = str(query.query_id)
            if qid in qrels:
                queries[qid] = query.text

        # Load passages (same as MS MARCO)
        print(f"   [Loading] passages (target: {max_docs:,})...")
        main_dataset = ir_datasets.load("msmarco-passage")
        corpus: Dict[str, Dict] = {}

        try:
            docstore = main_dataset.docs_store()
            for did in relevant_docs:
                try:
                    doc = docstore.get(did)
                    corpus[did] = {"text": doc.text, "title": ""}
                except Exception:
                    pass

            remaining = max_docs - len(corpus)
            if remaining > 0:
                count = 0
                for doc in main_dataset.docs_iter():
                    did = str(doc.doc_id)
                    if did not in corpus:
                        corpus[did] = {"text": doc.text, "title": ""}
                        count += 1
                        if count >= remaining:
                            break
        except Exception:
            count = 0
            for doc in main_dataset.docs_iter():
                did = str(doc.doc_id)
                corpus[did] = {"text": doc.text, "title": ""}
                count += 1
                if count >= max_docs:
                    break

        print(f"   [Loaded] {len(corpus):,} passages, {len(queries):,} queries")

        corpus_texts, corpus_ids, query_texts, query_ids = (
            self._build_ordered_lists(corpus, queries)
        )

        return BenchmarkData(
            name=f"TREC-DL-{year}",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            query_texts=query_texts,
            query_ids=query_ids,
        )


# ═══════════════════════════════════════════════════════════════════════
# Registry & convenience function
# ═══════════════════════════════════════════════════════════════════════

DATASET_REGISTRY = {
    "synthetic": SyntheticLoader,
    "beir": BEIRLoader,
    "msmarco": MSMARCOLoader,
    "trec-dl": TRECDLLoader,
}


def load_benchmark(
    source: str,
    **kwargs,
) -> BenchmarkData:
    """
    Unified entry point for loading any benchmark.

    Args:
        source: 'synthetic', 'beir', 'msmarco', or 'trec-dl'.
        **kwargs: passed to the specific loader.

    Examples:
        # Synthetic
        data = load_benchmark("synthetic", data_dir="ahrc_data")

        # BEIR SciFact
        data = load_benchmark("beir", dataset_name="scifact")

        # BEIR FiQA (subsampled)
        data = load_benchmark("beir", dataset_name="fiqa", max_docs=10000, max_queries=200)

        # MS MARCO dev small
        data = load_benchmark("msmarco", variant="dev/small", max_docs=50000, max_queries=500)

        # TREC DL 2019
        data = load_benchmark("trec-dl", year=2019, max_docs=50000)
    """
    if source not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset source: '{source}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    loader = DATASET_REGISTRY[source]()
    data = loader.load(**kwargs)
    print(data.summary())
    return data


def list_available_datasets():
    """Print all available datasets."""
    print("\n📚 Available Datasets:")
    print("=" * 60)

    print("\n  1. synthetic")
    print("     Your custom AHRC benchmark")

    print("\n  2. beir — BEIR Information Retrieval Benchmark")
    for name, desc in BEIR_DATASETS.items():
        print(f"     • {name}: {desc}")

    print("\n  3. msmarco — MS MARCO Passage Ranking")
    print("     • dev/small: 6,980 queries (recommended)")
    print("     • dev: full dev set")

    print("\n  4. trec-dl — TREC Deep Learning Track")
    print("     • year=2019: 43 hard queries with graded relevance")
    print("     • year=2020: 54 hard queries with graded relevance")

    print("\n" + "=" * 60)
