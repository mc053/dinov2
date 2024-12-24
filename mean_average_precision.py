# See https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

from typing import List
from decimal import Decimal, ROUND_HALF_UP

class PrecisionCalculator:
    def calculate_precision(self, relevant_documents: List[str], retrieved_documents: List[str]) -> float:
        relevant_and_retrieved = set(relevant_documents) & set(retrieved_documents)
        precision = len(relevant_and_retrieved) / len(retrieved_documents)

        return precision

class AveragePrecisionCalculator:
    def __init__(self, precision_calculator: PrecisionCalculator):
        self.precision_calculator = precision_calculator

    def calculate_average_precision(self, relevant_documents: List[str], retrieved_documents: List[str]) -> float:
        average_precision = 0.0
        found_relevant_documents = set()

        for k in range(1, len(retrieved_documents) + 1):
            top_k_documents = retrieved_documents[:k]
            precision_at_k = self.precision_calculator.calculate_precision(relevant_documents, top_k_documents)

            if retrieved_documents[k - 1] in relevant_documents:
                average_precision += precision_at_k
                found_relevant_documents.add(retrieved_documents[k - 1])

            if len(found_relevant_documents) == len(relevant_documents): # don't need to iterate further.
                break

        return average_precision / len(relevant_documents)

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
    
    
test_precision_calculator()
test_average_precision_calculator()