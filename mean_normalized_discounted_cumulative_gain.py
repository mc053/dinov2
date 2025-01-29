import numpy as np
from sklearn.metrics import ndcg_score

# Example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain

# [D1, D2, D3, D4, D5, D6, D7, D8]
y_true = np.array([[3, 2, 3, 0, 1, 2, 3, 2]])
y_pred = np.array([[8, 7, 6, 5, 4, 3, 2, 1]])

ndcg = ndcg_score(y_true, y_pred, k=6)

print(f"nDCG@6: {ndcg:.4f}")