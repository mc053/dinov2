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
        self.precision_calculator = precision_calculator
        self.hits = []

    def calculate_average_precision(self, relevant_documents: List[str], retrieved_documents: List[str]) -> float:
        average_precision = 0.0
        found_relevant_documents = set()
        self.hits = []

        for k in range(1, len(retrieved_documents) + 1):
            top_k_documents = retrieved_documents[:k]
            precision_at_k = self.precision_calculator.calculate_precision(relevant_documents, top_k_documents)

            if retrieved_documents[k - 1] in relevant_documents: # Hit
                self.hits.append(k)
                average_precision += precision_at_k
                found_relevant_documents.add(retrieved_documents[k - 1])

            if len(found_relevant_documents) == len(relevant_documents): # don't need to iterate further.
                break

        return average_precision / len(relevant_documents)

    # Maybe for plotting hit curves later.
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

# parser = argparse.ArgumentParser()
# 
# parser.add_argument("--gt")
# parser.add_argument("--percent", type=float)
# parser.add_argument("--query")
# parser.add_argument("--database")
# 
# args = parser.parse_args()
# 
# gt_retrievals_name = args.gt
# percent = args.percent
# query_embeddings_name = args.query
# database_embeddings_name = args.database
# 
# gt_retrievals_file = f"./retrieval_ground_truths/{gt_retrievals_name}"
# with open(gt_retrievals_file, "rb") as f:
#     print("Loading gt_retrievals_file...")
#     gt_retrievals = pickle.load(f)
#     print("gt_retrievals_file loaded.")
# 
# query_embeddings_file = f"./embeddings/{query_embeddings_name}"
# database_embeddings_file = f"./embeddings/{database_embeddings_name}"
# 
# query_embedding_names, query_embedding_vectors = load_embeddings(query_embeddings_file)
# database_embedding_names, database_embedding_vectors = load_embeddings(database_embeddings_file)
# 
# ap_calculator = AveragePrecisionCalculator(PrecisionCalculator())
# all_aps = []
# 
# for query_idx, query_embedding_name in tqdm(enumerate(query_embedding_names), total=len(query_embedding_names), desc="Calculating MAP"):
#     if query_embedding_name not in gt_retrievals:
#         raise ValueError(f"Query {query_embedding_name} not found in ground truth retrievals.")
#     
#     ground_truth = gt_retrievals[query_embedding_name]
#     relevant_count = int(len(ground_truth) * percent / 100)
#     relevant_documents = [list(d.keys())[0] for d in ground_truth[:relevant_count]]
#     
#     query_embedding_vector = query_embedding_vectors[query_idx].reshape(1, -1)
#     similarities = cosine_similarity(query_embedding_vector, database_embedding_vectors).flatten()    
#     
#     sorted_indices = np.argsort(-similarities)
#     retrieved_documents = [database_embedding_names[idx] for idx in sorted_indices]   
#     
#     ap = ap_calculator.calculate_average_precision(relevant_documents, retrieved_documents)
#     all_aps.append(ap)
# 
# map_score = np.mean(all_aps)
# 
# output_file = f"./retrieval_evaluations/map_{int(percent)}%_{query_embeddings_name}->{database_embeddings_name}.txt"
# with open(output_file, "w") as f:
#     f.write(f"MAP: {map_score}\n")
# 
# print(f"MAP calculation complete. Result saved to {output_file}.")