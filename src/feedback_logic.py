import numpy as np
from typing import List

def apply_rocchio(
    query_vector: np.ndarray,
    pos_vectors: List[np.ndarray],
    neg_vectors: List[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15
) -> np.ndarray:
    """
    Rocchio:
    Q_new = alpha * Q_old + beta * mean(Pos_Docs) - gamma * mean(Neg_Docs)
    """
    if pos_vectors and len(pos_vectors) > 0:
        pos_mean = np.mean(pos_vectors, axis=0)
    else:
        pos_mean = np.zeros_like(query_vector)

    if neg_vectors and len(neg_vectors) > 0:
        neg_mean = np.mean(neg_vectors, axis=0)
    else:
        neg_mean = np.zeros_like(query_vector)
    
    updated_query = (alpha * query_vector) + (beta * pos_mean) - (gamma * neg_mean)
    
    norm = np.linalg.norm(updated_query)
    if norm > 1e-12:
        return (updated_query / norm).astype(np.float32)
    return updated_query.astype(np.float32)