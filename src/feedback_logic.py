from __future__ import annotations

from typing import Sequence

import numpy as np

from src.part2_utils import l2_normalize


def apply_rocchio(
    query_vector: np.ndarray,
    positive_vectors: Sequence[np.ndarray],
    negative_vectors: Sequence[np.ndarray] | None = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    updated = alpha * np.asarray(query_vector, dtype=np.float32)

    if positive_vectors:
        updated += beta * np.mean(np.asarray(positive_vectors, dtype=np.float32), axis=0)
    if negative_vectors:
        updated -= gamma * np.mean(np.asarray(negative_vectors, dtype=np.float32), axis=0)

    return l2_normalize(updated)


def apply_facet_weights(
    query_vector: np.ndarray,
    facet_weights: Sequence[float] | None,
) -> np.ndarray:
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if not facet_weights:
        return l2_normalize(query_vector)

    facet_weights = [float(weight) for weight in facet_weights]
    scale = np.ones_like(query_vector, dtype=np.float32)
    n_facets = max(len(facet_weights), 1)

    # The encoder has no explicit facet basis, so we approximate a facet vector by
    # scaling contiguous slices of the dense query representation.
    for idx, weight in enumerate(facet_weights):
        start = (query_vector.shape[0] * idx) // n_facets
        end = (query_vector.shape[0] * (idx + 1)) // n_facets
        scale[start:end] = weight

    return l2_normalize(query_vector * scale)
