# See https://en.wikipedia.org/wiki/Discounted_cumulative_gain
import argparse
import numpy as np
from typing import List
from sklearn.metrics import ndcg_score

def extract_retrieved_similarities(ground_truth, retrieved_documents: List[str]) -> List[float]:
    ground_truth_dict = {k: v for d in ground_truth for k, v in d.items()}
    return [ground_truth_dict[doc] for doc in retrieved_documents]

def calculate_ndcg(retrieved_similarities: List[float], k = None) -> float:
        y_true = np.array([retrieved_similarities])
        y_pred = np.array([list(range(len(retrieved_similarities), 0, -1))])

        if k:
            return ndcg_score(y_true, y_pred, k=k)
        else:
            return ndcg_score(y_true, y_pred)

def test_extract_retrieved_similarities():
    ground_truth = [{'1.jpg': 1.00}, {'2.jpg': 0.66}, {'3.jpg': 0.65}, {'4.jpg': 0.63}]

    retrieved_documents_1 = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    retrieved_documents_2 = ['1.jpg', '3.jpg', '2.jpg', '4.jpg']
    retrieved_documents_3 = ['4.jpg', '3.jpg', '2.jpg', '1.jpg']

    retrieved_similarities_1 = extract_retrieved_similarities(ground_truth, retrieved_documents_1)
    retrieved_similarities_2 = extract_retrieved_similarities(ground_truth, retrieved_documents_2)
    retrieved_similarities_3 = extract_retrieved_similarities(ground_truth, retrieved_documents_3)

    assert retrieved_similarities_1 == [1.00, 0.66, 0.65, 0.63], f"Expected [1.00, 0.66, 0.65, 0.63], got {retrieved_similarities_1}"
    assert retrieved_similarities_2 == [1.00, 0.65, 0.66, 0.63], f"Expected [1.00, 0.65, 0.66, 0.63], got {retrieved_similarities_2}"
    assert retrieved_similarities_3 == [0.63, 0.65, 0.66, 1.00], f"Expected [0.63, 0.65, 0.66, 1.00], got {retrieved_similarities_3}"

def test_calculate_ndcg():
    wikipedia_example = [3, 2, 3, 0, 1, 2, 3, 2]
    retrieved_similarities_1 = [1.00, 0.66, 0.65, 0.63] # perfect retrievals
    retrieved_similarities_2 = [1.00, 0.65, 0.66, 0.63] # slightly worse retrievals
    retrieved_similarities_3 = [0.63, 0.65, 0.66, 1.00] # worse retrievals

    wikipedia_example_ndcg_at_6 = round(calculate_ndcg(wikipedia_example, k=6), 3)
    retrieved_similarities_1_ndcg = calculate_ndcg(retrieved_similarities_1)
    retrieved_similarities_2_ndcg = calculate_ndcg(retrieved_similarities_2)
    retrieved_similarities_3_ndcg = calculate_ndcg(retrieved_similarities_3)
    print(f"retrieved_similarities_1_ndcg: {retrieved_similarities_1_ndcg}, retrieved_similarities_2_ndcg: {retrieved_similarities_2_ndcg}, retrieved_similarities_3_ndcg: {retrieved_similarities_3_ndcg}")

    assert wikipedia_example_ndcg_at_6 == 0.785, f"Expected 0.785, got {wikipedia_example_ndcg_at_6}"
    assert retrieved_similarities_1_ndcg > retrieved_similarities_2_ndcg > retrieved_similarities_3_ndcg

test_extract_retrieved_similarities()
test_calculate_ndcg()

# parser = argparse.ArgumentParser()
# 
# parser.add_argument("--gt")
# parser.add_argument("--query")
# parser.add_argument("--database")
# 
# args = parser.parse_args()
# 
# gt_retrievals_name = args.gt
# query_embeddings_name = args.query
# database_embeddings_name = args.database