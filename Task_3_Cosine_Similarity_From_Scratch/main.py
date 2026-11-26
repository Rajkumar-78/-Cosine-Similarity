import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(a, b) / denominator)

tests = [
    ([1, 2, 3], [2, 4, 6], 1.0),
    ([1, 0], [0, 1], 0.0),
    ([1, 0], [-1, 0], -1.0),
]
for a, b, expected in tests:
    ours = cosine_similarity(a, b)
    reference = float(sklearn_cosine_similarity([a], [b])[0, 0])
    print(f"{a} vs {b} -> {ours:.1f} | sklearn: {reference:.1f}")
    assert np.isclose(ours, expected)
    assert np.isclose(ours, reference)