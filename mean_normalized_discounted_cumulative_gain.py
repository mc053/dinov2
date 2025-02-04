# See https://en.wikipedia.org/wiki/Discounted_cumulative_gain
import argparse
import pickle
import json
import numpy as np
from typing import List
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from mean_average_precision import load_embeddings

def extract_retrieved_similarities(ground_truth, retrieved_documents: List[str]) -> List[float]:
    ground_truth_dict = {k: v for d in ground_truth for k, v in d.items()}
    return [ground_truth_dict[doc] for doc in retrieved_documents]

def calculate_ndcg(retrieved_similarities: List[float], k = None) -> float:
    # Since cosine similarity is in the range of [-1, 1] and ndcg_score() can't handle negative values, we need to move
    # the similarities such that the lowest similarity becomes 0 while the absolute distance between the values
    # remains the same.
    # Example: [1, 0.5, 0, -0.2] becomes [1.2, 0.7, 0.2, 0]
    min_sim = min(retrieved_similarities)
    if min_sim < 0:
        retrieved_similarities = [s - min_sim for s in retrieved_similarities]

    y_true = np.array([retrieved_similarities])
    y_pred = np.array([list(range(len(retrieved_similarities), 0, -1))])

    if k:
        return ndcg_score(y_true, y_pred, k=k)
    else:
        return ndcg_score(y_true, y_pred)

def test_extract_retrieved_similarities():
    ground_truth = [{'1.jpg': 1.00}, {'2.jpg': 0.66}, {'3.jpg': 0.65}, {'4.jpg': 0.63}, {'5.jpg': -0.1}]

    retrieved_documents_1 = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    retrieved_documents_2 = ['1.jpg', '3.jpg', '2.jpg', '4.jpg', '5.jpg']
    retrieved_documents_3 = ['5.jpg', '4.jpg', '3.jpg', '2.jpg', '1.jpg']

    retrieved_similarities_1 = extract_retrieved_similarities(ground_truth, retrieved_documents_1)
    retrieved_similarities_2 = extract_retrieved_similarities(ground_truth, retrieved_documents_2)
    retrieved_similarities_3 = extract_retrieved_similarities(ground_truth, retrieved_documents_3)

    assert retrieved_similarities_1 == [1.00, 0.66, 0.65, 0.63, -0.1], f"Expected [1.00, 0.66, 0.65, 0.63], got {retrieved_similarities_1}"
    assert retrieved_similarities_2 == [1.00, 0.65, 0.66, 0.63, -0.1], f"Expected [1.00, 0.65, 0.66, 0.63], got {retrieved_similarities_2}"
    assert retrieved_similarities_3 == [-0.1, 0.63, 0.65, 0.66, 1.00], f"Expected [0.63, 0.65, 0.66, 1.00], got {retrieved_similarities_3}"

def test_calculate_ndcg():
    wikipedia_example = [3, 2, 3, 0, 1, 2, 3, 2]
    retrieved_similarities_1 = [1.00, 0.66, 0.65, 0.63, -0.1] # perfect retrievals
    retrieved_similarities_2 = [1.00, 0.65, 0.66, 0.63, -0.1] # slightly worse retrievals
    retrieved_similarities_3 = [-0.1, 0.63, 0.65, 0.66, 1.00] # worst retrievals

    wikipedia_example_ndcg_at_6 = round(calculate_ndcg(wikipedia_example, k=6), 3)
    retrieved_similarities_1_ndcg = calculate_ndcg(retrieved_similarities_1)
    retrieved_similarities_2_ndcg = calculate_ndcg(retrieved_similarities_2)
    retrieved_similarities_3_ndcg = calculate_ndcg(retrieved_similarities_3)
    print(f"retrieved_similarities_1_ndcg: {retrieved_similarities_1_ndcg}, retrieved_similarities_2_ndcg: {retrieved_similarities_2_ndcg}, retrieved_similarities_3_ndcg: {retrieved_similarities_3_ndcg}")

    assert wikipedia_example_ndcg_at_6 == 0.785, f"Expected 0.785, got {wikipedia_example_ndcg_at_6}"
    assert retrieved_similarities_1_ndcg > retrieved_similarities_2_ndcg > retrieved_similarities_3_ndcg

# test_extract_retrieved_similarities()
# test_calculate_ndcg()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt")
    parser.add_argument("--percent", type=float) # used for trying different k's (in ten steps) until percent is reached.
    parser.add_argument("--query")
    parser.add_argument("--database")

    args = parser.parse_args()

    gt_retrievals_name = args.gt
    percent = args.percent
    query_embeddings_name = args.query
    database_embeddings_name = args.database

    if "RVL_CDIP" in gt_retrievals_name:
        rvl_cdip_files = [
            f"./retrieval_ground_truths/RVL_CDIP_retrieval_ground_truths_{i}.pkl" for i in range(1, 5)
        ]
        print(f"RVL_CDIP mnDCG Evaluation: Found {len(rvl_cdip_files)} ground truth retrieval chunks.")
        current_rvl_cdip_file_index = 0
        
        def load_rvl_cdip_gt_retrievals(index):
            with open(rvl_cdip_files[index], "rb") as f:
                print(f"Loading RVL_CDIP Ground Truths from: {rvl_cdip_files[index]}...")
                return pickle.load(f)
        
        gt_retrievals = load_rvl_cdip_gt_retrievals(current_rvl_cdip_file_index)
    else: # CelebA
        gt_retrievals_file = f"./retrieval_ground_truths/{gt_retrievals_name}"
        with open(gt_retrievals_file, "rb") as f:
            print("Loading gt_retrievals_file...")
            gt_retrievals = pickle.load(f)
            print("gt_retrievals_file loaded.")

    query_embeddings_file = f"./embeddings/{query_embeddings_name}"
    database_embeddings_file = f"./embeddings/{database_embeddings_name}"

    query_embedding_names, query_embedding_vectors = load_embeddings(query_embeddings_file)
    database_embedding_names, database_embedding_vectors = load_embeddings(database_embeddings_file)

    total_ndcgs = []
    ndcgs_at_k = {}

    for query_idx, query_embedding_name in tqdm(enumerate(query_embedding_names), total=len(query_embedding_names), desc="Calculating mnDCG"):
        if query_embedding_name not in gt_retrievals:
            if "RVL_CDIP" in gt_retrievals_name:
                current_rvl_cdip_file_index += 1  # load next chunk
                if current_rvl_cdip_file_index >= len(rvl_cdip_files):
                    raise ValueError(f"Query {query_embedding_name} not found in RVL_CDIP Ground Truth files.")
                gt_retrievals = load_rvl_cdip_gt_retrievals(current_rvl_cdip_file_index)
            else:
                raise ValueError(f"Query {query_embedding_name} not found in ground truth retrievals.")
        
        ground_truth = gt_retrievals[query_embedding_name]
        max_k = int(len(ground_truth) * percent / 100)
        
        query_embedding_vector = query_embedding_vectors[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding_vector, database_embedding_vectors).flatten()    
        
        sorted_indices = np.argsort(-similarities)
        retrieved_documents = [database_embedding_names[idx] for idx in sorted_indices]   
        
        retrieved_similarities = extract_retrieved_similarities(ground_truth, retrieved_documents)

        for k in np.linspace(max_k // 10, max_k, num=10, dtype=int):
            ndcg_at_k = calculate_ndcg(retrieved_similarities, k=k)
            if k not in ndcgs_at_k:
                ndcgs_at_k[k] = []
            ndcgs_at_k[k].append(ndcg_at_k)

        total_ndcg = calculate_ndcg(retrieved_similarities)
        total_ndcgs.append(total_ndcg)

    mndcg_total = np.mean(total_ndcgs)
    mndcg_at_k = {k: np.mean(values) for k, values in ndcgs_at_k.items()}

    mndcg_output_file = f"./retrieval_evaluations/mndcg_{query_embeddings_name}->{database_embeddings_name}.txt"
    with open(mndcg_output_file, "w") as f:
        for k, mndcg in mndcg_at_k.items():
            f.write(f"mnDCG@{k}: {mndcg}\n")
        f.write(f"Total mnDCG: {mndcg_total}\n")

    print(f"mnDCG calculation complete. Results saved to {mndcg_output_file}.")