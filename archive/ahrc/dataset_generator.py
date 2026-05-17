"""
AHRC — Dataset Generator
Produces synthetic benchmark with ground-truth relevance labels.

Each task has:
  - text description (from category templates)
  - metadata (category, priority, complexity, domain)
  - graph relationships (edges to semantically related tasks)

Each query has:
  - text + embedding
  - ground-truth relevance labels for all tasks (0–3 scale)
"""

import json
import random
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .config import AHRCConfig


# ── Category definitions ──────────────────────────────────────────────────

CATEGORIES = {
    "machine_learning": {
        "templates": [
            "Train a {} model for {} classification",
            "Optimize {} hyperparameters using {} search",
            "Implement {} regularization for {} network",
            "Evaluate {} performance on {} dataset",
            "Debug gradient {} in {} architecture",
            "Design {} loss function for {} task",
            "Benchmark {} vs {} on standard datasets",
        ],
        "modifiers_a": ["transformer", "CNN", "LSTM", "GAN", "diffusion", "VAE", "ResNet", "BERT"],
        "modifiers_b": ["image", "text", "audio", "tabular", "multimodal", "time-series", "graph"],
        "keywords": ["model", "training", "neural", "deep learning", "optimization"],
    },
    "data_engineering": {
        "templates": [
            "Build {} pipeline for {} ingestion",
            "Implement {} schema validation for {} sources",
            "Optimize {} query performance on {} tables",
            "Design {} partitioning strategy for {} data",
            "Create {} monitoring dashboard for {} pipeline",
            "Migrate {} storage from {} to cloud",
        ],
        "modifiers_a": ["ETL", "streaming", "batch", "real-time", "CDC", "event-driven"],
        "modifiers_b": ["customer", "transaction", "sensor", "log", "clickstream", "IoT"],
        "keywords": ["pipeline", "data", "ingestion", "schema", "ETL"],
    },
    "security": {
        "templates": [
            "Implement {} authentication for {} service",
            "Audit {} compliance in {} infrastructure",
            "Design {} encryption for {} at rest",
            "Build {} intrusion detection for {} network",
            "Create {} access control for {} resources",
        ],
        "modifiers_a": ["OAuth2", "mTLS", "zero-trust", "RBAC", "MFA", "SAML"],
        "modifiers_b": ["API", "microservice", "database", "storage", "messaging", "edge"],
        "keywords": ["security", "authentication", "encryption", "compliance", "access"],
    },
    "infrastructure": {
        "templates": [
            "Deploy {} cluster on {} platform",
            "Configure {} auto-scaling for {} workload",
            "Implement {} service mesh for {} topology",
            "Optimize {} resource allocation for {} services",
            "Design {} disaster recovery for {} region",
        ],
        "modifiers_a": ["Kubernetes", "Docker", "Terraform", "Ansible", "Helm", "ArgoCD"],
        "modifiers_b": ["production", "staging", "multi-region", "edge", "hybrid", "serverless"],
        "keywords": ["deployment", "infrastructure", "cluster", "scaling", "cloud"],
    },
    "nlp": {
        "templates": [
            "Build {} extraction pipeline for {} documents",
            "Implement {} summarization for {} content",
            "Design {} intent classifier for {} queries",
            "Create {} entity linker for {} knowledge base",
            "Optimize {} tokenizer for {} language",
        ],
        "modifiers_a": ["NER", "relation", "sentiment", "abstractive", "extractive", "semantic"],
        "modifiers_b": ["medical", "legal", "financial", "scientific", "social media", "news"],
        "keywords": ["NLP", "text", "language", "extraction", "summarization"],
    },
    "computer_vision": {
        "templates": [
            "Train {} detector for {} imagery",
            "Implement {} segmentation for {} analysis",
            "Build {} tracking system for {} video",
            "Design {} augmentation pipeline for {} data",
            "Optimize {} inference for {} deployment",
        ],
        "modifiers_a": ["YOLO", "Mask-RCNN", "ViT", "DETR", "SAM", "U-Net"],
        "modifiers_b": ["satellite", "medical", "autonomous", "industrial", "retail", "surveillance"],
        "keywords": ["vision", "image", "detection", "segmentation", "visual"],
    },
    "recommendation": {
        "templates": [
            "Build {} recommendation engine for {} items",
            "Implement {} collaborative filtering for {} users",
            "Design {} content-based ranker for {} catalog",
            "Optimize {} diversity in {} recommendations",
            "Create {} cold-start strategy for {} platform",
        ],
        "modifiers_a": ["hybrid", "neural", "graph-based", "multi-task", "contextual", "session-based"],
        "modifiers_b": ["product", "movie", "music", "news", "job", "course"],
        "keywords": ["recommendation", "ranking", "collaborative", "personalization"],
    },
    "optimization": {
        "templates": [
            "Implement {} optimizer for {} scheduling",
            "Design {} constraint solver for {} allocation",
            "Build {} search heuristic for {} routing",
            "Optimize {} convergence for {} objective",
            "Create {} meta-learning strategy for {} tuning",
        ],
        "modifiers_a": ["genetic", "Bayesian", "gradient-free", "simulated annealing", "PSO", "CMA-ES"],
        "modifiers_b": ["resource", "job", "vehicle", "portfolio", "network", "warehouse"],
        "keywords": ["optimization", "scheduling", "constraint", "search", "convergence"],
    },
    "testing": {
        "templates": [
            "Write {} tests for {} module",
            "Implement {} coverage analysis for {} codebase",
            "Build {} fuzzer for {} API endpoints",
            "Design {} regression suite for {} pipeline",
            "Create {} load test for {} service",
        ],
        "modifiers_a": ["unit", "integration", "property-based", "mutation", "contract", "chaos"],
        "modifiers_b": ["authentication", "payment", "search", "notification", "analytics", "core"],
        "keywords": ["testing", "test", "coverage", "regression", "quality"],
    },
    "database": {
        "templates": [
            "Optimize {} index strategy for {} queries",
            "Implement {} sharding for {} database",
            "Design {} migration plan for {} schema",
            "Build {} caching layer for {} access patterns",
            "Create {} replication topology for {} cluster",
        ],
        "modifiers_a": ["B-tree", "LSM", "columnar", "graph", "vector", "time-series"],
        "modifiers_b": ["OLTP", "OLAP", "real-time", "analytical", "transactional", "hybrid"],
        "keywords": ["database", "query", "index", "sharding", "cache"],
    },
    "frontend": {
        "templates": [
            "Build {} component library for {} design system",
            "Implement {} state management for {} application",
            "Optimize {} rendering for {} performance",
            "Design {} accessibility features for {} interface",
            "Create {} animation system for {} interactions",
        ],
        "modifiers_a": ["React", "Vue", "Svelte", "Web Component", "SSR", "Islands"],
        "modifiers_b": ["dashboard", "e-commerce", "mobile", "PWA", "SPA", "embedded"],
        "keywords": ["frontend", "UI", "component", "rendering", "interface"],
    },
    "mlops": {
        "templates": [
            "Build {} model registry for {} pipeline",
            "Implement {} experiment tracking for {} models",
            "Design {} feature store for {} serving",
            "Create {} A/B testing framework for {} models",
            "Optimize {} inference serving for {} latency",
        ],
        "modifiers_a": ["MLflow", "Kubeflow", "Vertex AI", "SageMaker", "custom", "open-source"],
        "modifiers_b": ["production", "research", "batch", "real-time", "multi-model", "edge"],
        "keywords": ["MLOps", "deployment", "model", "serving", "experiment"],
    },
}


@dataclass
class Task:
    """A single task in the benchmark corpus."""
    id: str
    description: str
    category: str
    priority: float
    complexity: float
    domain: str
    keywords: List[str]
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    neighbors: List[str] = field(default_factory=list)


@dataclass
class Query:
    """A benchmark query with ground-truth labels."""
    id: str
    text: str
    source_task_id: str
    category: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    relevance: Dict[str, int] = field(default_factory=dict)  # task_id → {0,1,2,3}


class DatasetGenerator:
    """Generate synthetic retrieval benchmark."""

    def __init__(self, config: AHRCConfig):
        self.cfg = config.experiment
        self.rng = random.Random(self.cfg.random_seed)
        self.np_rng = np.random.default_rng(self.cfg.random_seed)
        self.tasks: List[Task] = []
        self.queries: List[Query] = []
        self.category_names = list(CATEGORIES.keys())[:self.cfg.num_categories]

    # ── Task generation ────────────────────────────────────────────────

    def generate_tasks(self) -> List[Task]:
        """Generate N tasks with category-based descriptions."""
        print(f"📦 Generating {self.cfg.num_tasks:,} tasks across {len(self.category_names)} categories...")
        t0 = time.time()
        tasks = []

        for i in range(self.cfg.num_tasks):
            cat_name = self.rng.choice(self.category_names)
            cat = CATEGORIES[cat_name]

            template = self.rng.choice(cat["templates"])
            mod_a = self.rng.choice(cat["modifiers_a"])
            mod_b = self.rng.choice(cat["modifiers_b"])
            description = template.format(mod_a, mod_b)

            # Add noise words for realism
            if self.rng.random() < 0.3:
                noise = self.rng.choice([
                    " with monitoring", " using CI/CD", " in production",
                    " with logging", " for v2 release", " urgently",
                    " for Q3 deadline", " as prototype", " with tests",
                ])
                description += noise

            task = Task(
                id=f"task_{i:06d}",
                description=description,
                category=cat_name,
                priority=round(self.rng.uniform(0.1, 1.0), 3),
                complexity=round(self.rng.uniform(0.1, 1.0), 3),
                domain=mod_b,
                keywords=cat["keywords"][:] + [mod_a.lower(), mod_b.lower()],
            )
            tasks.append(task)

        self.tasks = tasks
        elapsed = time.time() - t0
        print(f"   ✅ Generated {len(tasks):,} tasks in {elapsed:.2f}s")
        return tasks

    # ── Graph construction ─────────────────────────────────────────────

    def build_graph(self) -> int:
        """Create task relationships based on category + random cross-links."""
        print("🔗 Building task relationship graph...")
        t0 = time.time()

        # Group by category
        cat_groups: Dict[str, List[Task]] = defaultdict(list)
        for t in self.tasks:
            cat_groups[t.category].append(t)

        edge_count = 0
        target_edges = int(self.cfg.num_tasks * self.cfg.avg_relationships_per_task / 2)

        # Intra-category edges (70% of target)
        intra_target = int(target_edges * 0.70)
        for _ in range(intra_target):
            cat = self.rng.choice(self.category_names)
            group = cat_groups[cat]
            if len(group) < 2:
                continue
            a, b = self.rng.sample(group, 2)
            if b.id not in a.neighbors and a.id not in b.neighbors:
                a.neighbors.append(b.id)
                b.neighbors.append(a.id)
                edge_count += 1

        # Cross-category edges (30% of target)
        cross_target = target_edges - edge_count
        for _ in range(cross_target):
            a = self.rng.choice(self.tasks)
            b = self.rng.choice(self.tasks)
            if a.id != b.id and b.id not in a.neighbors:
                a.neighbors.append(b.id)
                b.neighbors.append(a.id)
                edge_count += 1

        elapsed = time.time() - t0
        avg_deg = np.mean([len(t.neighbors) for t in self.tasks])
        print(f"   ✅ {edge_count:,} edges, avg degree {avg_deg:.1f}, in {elapsed:.2f}s")
        return edge_count

    # ── Embedding generation ───────────────────────────────────────────

    def generate_embeddings(self) -> None:
        """Embed all task descriptions using SentenceTransformer."""
        print(f"🧠 Generating embeddings ({self.cfg.model_name})...")
        t0 = time.time()

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.cfg.model_name, device='cpu')
        descriptions = [t.description for t in self.tasks]

        # Batch encode
        embeddings = model.encode(
            descriptions,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        for task, emb in zip(self.tasks, embeddings):
            task.embedding = emb.astype(np.float32)

        self.cfg_embedding_dim = embeddings.shape[1]
        elapsed = time.time() - t0
        print(f"   ✅ Embedded {len(self.tasks):,} tasks (dim={embeddings.shape[1]}) in {elapsed:.1f}s")

    # ── Query + ground-truth generation ────────────────────────────────

    def generate_queries(self) -> List[Query]:
        """Generate queries with multi-level relevance labels."""
        print(f"🔍 Generating {self.cfg.num_queries} queries with ground-truth labels...")
        t0 = time.time()

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.cfg.model_name, device='cpu')

        # Build task-id lookup
        task_map = {t.id: t for t in self.tasks}

        # Category index for relevance scoring
        cat_tasks: Dict[str, List[Task]] = defaultdict(list)
        for t in self.tasks:
            cat_tasks[t.category].append(t)

        queries = []

        # Select source tasks for queries
        source_indices = self.rng.sample(range(len(self.tasks)), min(self.cfg.num_queries, len(self.tasks)))

        for qi, src_idx in enumerate(source_indices):
            src = self.tasks[src_idx]

            # Paraphrase the source task slightly
            query_text = self._paraphrase(src.description)

            query = Query(
                id=f"query_{qi:04d}",
                text=query_text,
                source_task_id=src.id,
                category=src.category,
            )

            # Encode query
            query.embedding = model.encode(
                query_text, normalize_embeddings=True
            ).astype(np.float32)

            # Assign ground-truth relevance
            query.relevance = self._compute_relevance(src, task_map, cat_tasks)
            queries.append(query)

        self.queries = queries
        elapsed = time.time() - t0
        avg_relevant = np.mean([
            sum(1 for v in q.relevance.values() if v >= 2)
            for q in queries
        ])
        print(f"   ✅ {len(queries)} queries, avg {avg_relevant:.1f} relevant tasks/query, in {elapsed:.1f}s")
        return queries

    def _paraphrase(self, text: str) -> str:
        """Lightweight paraphrase: word swap, prefix change."""
        transforms = [
            lambda t: "Find tasks related to: " + t,
            lambda t: t.replace("Build", "Construct").replace("Implement", "Develop"),
            lambda t: t.replace("Design", "Architect").replace("Optimize", "Improve"),
            lambda t: "Need help with: " + t.lower(),
            lambda t: t + " — looking for similar work",
        ]
        return self.rng.choice(transforms)(text)

    def _compute_relevance(
        self,
        src: Task,
        task_map: Dict[str, Task],
        cat_tasks: Dict[str, List[Task]],
    ) -> Dict[str, int]:
        """
        Multi-level relevance scoring:
          3 = highly relevant (graph neighbor + same category)
          2 = relevant (same category OR graph neighbor)
          1 = marginal (adjacent category or shared keyword)
          0 = irrelevant
        """
        relevance = {}
        neighbor_set = set(src.neighbors)
        src_kw = set(src.keywords)

        for t in self.tasks:
            if t.id == src.id:
                continue

            same_cat = (t.category == src.category)
            is_neighbor = (t.id in neighbor_set)
            keyword_overlap = len(src_kw.intersection(set(t.keywords))) >= 2

            if same_cat and is_neighbor:
                relevance[t.id] = 3
            elif same_cat or is_neighbor:
                relevance[t.id] = 2
            elif keyword_overlap:
                relevance[t.id] = 1
            else:
                relevance[t.id] = 0

        return relevance

    # ── Serialization ──────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Save dataset to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Tasks (without embeddings — too large for JSON)
        tasks_data = []
        for t in self.tasks:
            tasks_data.append({
                "id": t.id,
                "description": t.description,
                "category": t.category,
                "priority": t.priority,
                "complexity": t.complexity,
                "domain": t.domain,
                "keywords": t.keywords,
                "neighbors": t.neighbors,
            })

        with open(os.path.join(output_dir, "tasks.json"), "w") as f:
            json.dump(tasks_data, f, indent=2)

        # Embeddings as numpy array
        embeddings = np.array([t.embedding for t in self.tasks])
        np.save(os.path.join(output_dir, "task_embeddings.npy"), embeddings)

        # Queries
        queries_data = []
        for q in self.queries:
            queries_data.append({
                "id": q.id,
                "text": q.text,
                "source_task_id": q.source_task_id,
                "category": q.category,
                "relevance": q.relevance,
            })

        with open(os.path.join(output_dir, "queries.json"), "w") as f:
            json.dump(queries_data, f, indent=2)

        # Query embeddings
        query_embeddings = np.array([q.embedding for q in self.queries])
        np.save(os.path.join(output_dir, "query_embeddings.npy"), query_embeddings)

        # Metadata
        meta = {
            "num_tasks": len(self.tasks),
            "num_queries": len(self.queries),
            "num_categories": len(self.category_names),
            "categories": self.category_names,
            "embedding_dim": self.tasks[0].embedding.shape[0] if self.tasks else 0,
            "avg_degree": float(np.mean([len(t.neighbors) for t in self.tasks])),
            "total_edges": sum(len(t.neighbors) for t in self.tasks) // 2,
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"💾 Dataset saved to {output_dir}/")

    @classmethod
    def load(cls, output_dir: str, config: AHRCConfig) -> "DatasetGenerator":
        """Load pre-generated dataset from disk."""
        gen = cls(config)

        with open(os.path.join(output_dir, "tasks.json")) as f:
            tasks_data = json.load(f)

        embeddings = np.load(os.path.join(output_dir, "task_embeddings.npy"))

        gen.tasks = []
        for td, emb in zip(tasks_data, embeddings):
            task = Task(
                id=td["id"],
                description=td["description"],
                category=td["category"],
                priority=td["priority"],
                complexity=td["complexity"],
                domain=td["domain"],
                keywords=td["keywords"],
                embedding=emb.astype(np.float32),
                neighbors=td["neighbors"],
            )
            gen.tasks.append(task)

        with open(os.path.join(output_dir, "queries.json")) as f:
            queries_data = json.load(f)

        query_embeddings = np.load(os.path.join(output_dir, "query_embeddings.npy"))

        gen.queries = []
        for qd, emb in zip(queries_data, query_embeddings):
            query = Query(
                id=qd["id"],
                text=qd["text"],
                source_task_id=qd["source_task_id"],
                category=qd["category"],
                embedding=emb.astype(np.float32),
                relevance=qd["relevance"],
            )
            gen.queries.append(query)

        print(f"📂 Loaded dataset: {len(gen.tasks):,} tasks, {len(gen.queries)} queries from {output_dir}/")
        return gen


def build_dataset(config: AHRCConfig, output_dir: str = "ahrc_data") -> DatasetGenerator:
    """Full pipeline: generate tasks → graph → embeddings → queries → save."""
    gen = DatasetGenerator(config)
    gen.generate_tasks()
    gen.build_graph()
    gen.generate_embeddings()
    gen.generate_queries()
    gen.save(output_dir)
    return gen


if __name__ == "__main__":
    config = AHRCConfig()
    build_dataset(config)
