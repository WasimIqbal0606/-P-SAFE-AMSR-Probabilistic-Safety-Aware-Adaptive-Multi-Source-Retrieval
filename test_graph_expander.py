import numpy as np
from archive.ahrc.graph_expander import GraphExpander
from archive.ahrc.config import AdaptiveConfig

def test_graph_expander_adjacency():
    cfg = AdaptiveConfig()
    graph_exp = GraphExpander(cfg)
    
    # Create fake embeddings
    embeddings = np.random.rand(10, 128).astype(np.float32)
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build graph
    import faiss
    idx = faiss.IndexFlatIP(128)
    idx.add(embeddings)
    _, indices = idx.search(embeddings, 3) # k=2 neighbors
    
    graph_exp.adjacency = {}
    graph_exp._built = True
    for i in range(10):
        neighbors = [int(indices[i, j]) for j in range(1, 3) if 0 <= indices[i, j] < 10]
        graph_exp.adjacency[i] = set(neighbors)
        graph_exp.degree_cache[i] = len(neighbors)
        
    # Simulate dense retrieval giving node 0
    seed_indices = np.array([0])
    seed_scores = np.array([1.0])
    
    query_emb = embeddings[0]
    
    # Expand
    exp_indices, exp_scores = graph_exp.expand(
        seed_indices, seed_scores, query_emb, embeddings, hops=1, max_neighbors=5
    )
    
    # Assert graph-only candidates are returned
    assert len(exp_indices) > len(seed_indices), "No new candidates were added by graph expansion."
    
    graph_only = set(exp_indices) - set(seed_indices)
    assert len(graph_only) > 0, "No graph-only candidates were found."
    
    print("GraphExpander unit test passed! Graph-only candidates successfully returned.")

if __name__ == "__main__":
    test_graph_expander_adjacency()
