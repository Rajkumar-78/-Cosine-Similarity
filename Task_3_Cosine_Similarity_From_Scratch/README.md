# Task 3 — Cosine Similarity from Scratch

## Run
`pip install -r requirements.txt`, then run `notebook.ipynb`.

## Expected output
- `[1, 2, 3] vs [2, 4, 6] -> 1.0`
- `[1, 0] vs [0, 1] -> 0.0`
- `[1, 0] vs [-1, 0] -> -1.0`

Each result is also compared with `sklearn.metrics.pairwise.cosine_similarity`. The notebook verifies that cosine similarity equals the dot product after L2-normalising both vectors.
