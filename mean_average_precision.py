# See https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
import argparse
import pickle
import json
import numpy as np
from typing import List
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from decimal import Decimal, ROUND_HALF_UP

class PrecisionCalculator:
    def calculate_precision(self, relevant_documents: List[str], retrieved_documents: List[str]) -> float:
        relevant_and_retrieved = set(relevant_documents) & set(retrieved_documents)
        precision = len(relevant_and_retrieved) / len(retrieved_documents)

        return precision

class AveragePrecisionCalculator:
    def __init__(self, precision_calculator: PrecisionCalculator):
        # Not needed anymore.
        # self.precision_calculator = precision_calculator
        self.hits = []

    def calculate_average_precision(self, relevant_documents: List[str], retrieved_documents: List[str]) -> float:
        relevant_set = set(relevant_documents)
        average_precision = 0.0
        found_relevant_count = 0
        self.hits = []

        relevant_indices = [idx for idx, doc in enumerate(retrieved_documents) if doc in relevant_set]

        for rank, idx in enumerate(relevant_indices, start=1):
            precision_at_k = rank / (idx + 1)
            average_precision += precision_at_k
            self.hits.append(idx + 1)
 
            found_relevant_count += 1
            if found_relevant_count == len(relevant_documents):
                break

        return average_precision / len(relevant_documents)

    # Call this method after calling calculate_average_precision - Might be useful for plotting hit curves later.
    def get_hits(self) -> list:
        return self.hits

def test_precision_calculator():
    calculator = PrecisionCalculator()
    relevant = ['001.jpg', '002.jpg', '003.jpg']

    assert calculator.calculate_precision(relevant, ['001.jpg']) == 1.0, "Should be 1.0"
    assert calculator.calculate_precision(relevant, ['002.jpg']) == 1.0, "Should be 1.0"
    assert calculator.calculate_precision(relevant, ['003.jpg']) == 1.0, "Should be 1.0"

    assert calculator.calculate_precision(relevant, ['001.jpg', '002.jpg']) == 1.0, "Should be 1.0"
    assert calculator.calculate_precision(relevant, ['002.jpg', '004.jpg']) == 1/2, "Should be 1/2"
    assert calculator.calculate_precision(relevant, ['004.jpg', '005.jpg']) == 0.0, "Should be 0.0"

# Test based on page 7: https://datascience-intro.github.io/1MS041-2022/Files/AveragePrecision.pdf
def test_average_precision_calculator():
    calculator = AveragePrecisionCalculator(PrecisionCalculator())
    relevant = ['001.jpg', '002.jpg', '003.jpg']
   
    algorithm_a_retrievals = ['001.jpg', '002.jpg', '004.jpg', '003.jpg', '005.jpg', '006.jpg', '007.jpg', '008.jpg', '009.jpg', '010.jpg']
    algorithm_b_retrievals = ['001.jpg', '004.jpg', '005.jpg', '002.jpg', '006.jpg', '007.jpg', '008.jpg', '003.jpg', '009.jpg', '010.jpg']
    
    ap_a = calculator.calculate_average_precision(relevant, algorithm_a_retrievals)
    ap_b = calculator.calculate_average_precision(relevant, algorithm_b_retrievals)
    
    rounded_ap_a = float(Decimal(ap_a).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    rounded_ap_b = float(Decimal(ap_b).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    
    assert rounded_ap_a == 0.92, f"Expected 0.92, got {rounded_ap_a}"
    assert rounded_ap_b == 0.63, f"Expected 0.63, got {rounded_ap_b}"

# test_precision_calculator()
# test_average_precision_calculator()

def load_embeddings(file_path):
    with open(file_path, "r") as f:
        embeddings_data = json.load(f)

    embedding_names = list(embeddings_data.keys())
    embedding_vectors = np.array([embeddings_data[name] for name in embedding_names])

    return embedding_names, embedding_vectors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt")
    parser.add_argument("--percent", type=float)
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
        print(f"RVL_CDIP mAP Evaluation: Found {len(rvl_cdip_files)} ground truth retrieval chunks.")
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

    ap_calculator = AveragePrecisionCalculator(PrecisionCalculator())
    all_aps = []
    all_hits = {}

    for query_idx, query_embedding_name in tqdm(enumerate(query_embedding_names), total=len(query_embedding_names), desc="Calculating mAP"):
        if query_embedding_name not in gt_retrievals:
            if "RVL_CDIP" in gt_retrievals_name:
                current_rvl_cdip_file_index += 1  # load next chunk
                if current_rvl_cdip_file_index >= len(rvl_cdip_files):
                    raise ValueError(f"Query {query_embedding_name} not found in RVL_CDIP Ground Truth files.")
                gt_retrievals = load_rvl_cdip_gt_retrievals(current_rvl_cdip_file_index)
            else:
                raise ValueError(f"Query {query_embedding_name} not found in ground truth retrievals.")
        
        ground_truth = gt_retrievals[query_embedding_name]
        relevant_count = int(len(ground_truth) * percent / 100)
        relevant_documents = [list(d.keys())[0] for d in ground_truth[:relevant_count]]
        
        query_embedding_vector = query_embedding_vectors[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding_vector, database_embedding_vectors).flatten()    
        
        sorted_indices = np.argsort(-similarities)
        retrieved_documents = [database_embedding_names[idx] for idx in sorted_indices]   
        
        ap = ap_calculator.calculate_average_precision(relevant_documents, retrieved_documents)
        hits = ap_calculator.get_hits()

        all_aps.append(ap)
        all_hits[query_embedding_name] = hits

    map_score = np.mean(all_aps)

    map_output_file = f"./retrieval_evaluations/map_{int(percent)}%_{query_embeddings_name}->{database_embeddings_name}.txt"
    hits_output_file = f"./retrieval_evaluations/hits_{int(percent)}%_{query_embeddings_name}->{database_embeddings_name}.json"

    with open(map_output_file, "w") as f:
        f.write(f"mAP: {map_score}\n")

    with open(hits_output_file, "w") as f:
        json.dump(all_hits, f)

    print(f"mAP calculation complete. Results saved to {map_output_file} and {hits_output_file}.")