"""Demonstrate online mapping of new students during inference."""
import numpy as np


def map_query_to_targets(query_embedding, target_embeddings, target_ids, top_k=3):
    """Map a query embedding to top-k targets with similarity scores."""
    dists = np.linalg.norm(target_embeddings - query_embedding, axis=1)
    idx = np.argsort(dists)[:top_k]
    return [(target_ids[i], float(1 / (1 + dists[i]))) for i in idx]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    targets = rng.normal(size=(10, 32))
    ids = [str(i) for i in range(10)]
    query = rng.normal(size=(32,))
    print(map_query_to_targets(query, targets, ids, top_k=3))
