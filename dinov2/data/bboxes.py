import os
import csv
import json
import random
import numpy as np
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from mtcnn.mtcnn import MTCNN
from typing import Optional, Tuple
from tqdm import tqdm

# Adjust main method before starting job.

class CelebABBoxDetector:
    def detect_celeba_bbox(self, img_path) -> Optional[Tuple[int, int, int, int]]:
        raise NotImplementedError

class CelebABBoxDetectorMTCNN(CelebABBoxDetector):
    def __init__(self):
        self.detector = MTCNN()

    def detect_celeba_bbox(self, img_path: str) -> Optional[Tuple[int, int, int, int]]:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None

        img_array = np.asarray(img)
        detections = self.detector.detect_faces(img_array)

        if not detections:
            print(f"No faces detected in {img_path}")
            return None

        bbox = detections[0]["box"]
        x_1, y_1, width, height = bbox
        x_1 = max(0, x_1)
        y_1 = max(0, y_1)

        return x_1, y_1, width, height

class RvlCdipBBoxesDetector:
    def detect_rvl_cdip_bboxes(self, img_path) -> Optional[list]:
        raise NotImplementedError

class RvlCdipBBoxesDetectorPaddleOCR(RvlCdipBBoxesDetector):
    def __init__(self):
        self.detector = PaddleOCR(use_angle_cls=True, lang='en')

    def detect_rvl_cdip_bboxes(self, img_path) -> Optional[list]:
        try:
            result = self.detector.ocr(img_path, rec=False)
            bboxes = []
            for line in result[0]:
                bboxes.append(line)
            return bboxes
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

def save_celeba_bboxes_to_csv(detector: CelebABBoxDetectorMTCNN, image_dir: str, output_csv: str):
    images = sorted(os.listdir(image_dir))
    output_path = os.path.abspath(output_csv)

    with open(output_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_id", "x_1", "y_1", "width", "height"])

        for image in tqdm(images, desc="Processing images"):
            img_path = os.path.join(image_dir, image)
            bbox = detector.detect_celeba_bbox(img_path)

            if bbox:
                csvwriter.writerow([image, *bbox])

def save_rvl_cdip_bboxes_to_json(detector: RvlCdipBBoxesDetectorPaddleOCR, image_dir: str, output_json: str):
    images = os.listdir(image_dir)
    output_path = os.path.abspath(output_json)

    output_data = {}

    for image in tqdm(images, desc="Processing RVL-CDIP images"):
        img_path = os.path.join(image_dir, image)

        if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        bboxes = detector.detect_rvl_cdip_bboxes(img_path)
        if bboxes:
            output_data[image] = {
                "original_bbs_count": len(bboxes),
                "bbs": bboxes
            }

    with open(output_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"BBoxes saved to {output_json}")

REDUCTION_FRACTION = 0.25

class ReductionStep(Enum):
    FROM_100_TO_75 = 1
    FROM_75_TO_50 = 2
    FROM_50_TO_25 = 3

def round_half_up(value: float) -> int:
    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def calculate_reduction_amount(original_count: int, reduction_step: ReductionStep):
    # Edge Case
    if original_count == 1:
        return 0

    base_reduction = original_count * REDUCTION_FRACTION
    remainder = base_reduction % 1

    if remainder == 0.75:
        if reduction_step == ReductionStep.FROM_100_TO_75:
            return round_half_up(base_reduction)  # Round up
        elif reduction_step == ReductionStep.FROM_75_TO_50:
            return int(base_reduction)  # Round off
        elif reduction_step == ReductionStep.FROM_50_TO_25:
            return round_half_up(base_reduction)  # Round up

    elif remainder == 0.5:
        if reduction_step == ReductionStep.FROM_100_TO_75:
            return int(base_reduction)  # Round off
        elif reduction_step == ReductionStep.FROM_75_TO_50:
            return round_half_up(base_reduction)  # Round up
        elif reduction_step == ReductionStep.FROM_50_TO_25:
            return int(base_reduction)  # Round off

    elif remainder == 0.25:
        if reduction_step == ReductionStep.FROM_100_TO_75:
            return int(base_reduction)  # Round off
        elif reduction_step == ReductionStep.FROM_75_TO_50:
            return int(base_reduction)  # Round off
        elif reduction_step == ReductionStep.FROM_50_TO_25:
            return int(base_reduction) + 1  # Round off and add 1
    
    return base_reduction

def test_calculate_reduction_amount():
    original_count = 103                                                                    # bbs left:
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_100_TO_75) == 26   # 77
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_75_TO_50) == 25    # 52
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_50_TO_25) == 26    # 26

    original_count = 102
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_100_TO_75) == 25   # 77
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_75_TO_50) == 26    # 51
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_50_TO_25) == 25    # 26

    original_count = 101
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_100_TO_75) == 25   # 76
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_75_TO_50) == 25    # 51
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_50_TO_25) == 26    # 25

    original_count = 3
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_100_TO_75) == 1    # 2
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_75_TO_50) == 0     # 2
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_50_TO_25) == 1     # 1

    original_count = 2
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_100_TO_75) == 0    # 2
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_75_TO_50) == 1     # 1
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_50_TO_25) == 0     # 1

    # Edge Case
    original_count = 1
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_100_TO_75) == 0    # 1
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_75_TO_50) == 0     # 1
    assert calculate_reduction_amount(original_count, ReductionStep.FROM_50_TO_25) == 0     # 1

    print("All tests passed.")

def reduce_rvl_cdip_bboxes(input_json: str, output_json: str, reduction_step: ReductionStep):
    with open(input_json, "r") as json_file:
        data = json.load(json_file)

    reduced_data = {}

    for image_name, content in tqdm(data.items(), desc="Reducing Bounding Boxes"):
        bbs = content["bbs"]
        original_bbs_count = content["original_bbs_count"]

        reduction_amount = calculate_reduction_amount(original_bbs_count, reduction_step)
        reduced_count = int(len(bbs) - reduction_amount)
        reduced_bbs = random.sample(bbs, reduced_count)

        reduced_data[image_name] = {
            "original_bbs_count": original_bbs_count,
            "bbs": reduced_bbs
        }

    with open(output_json, "w") as output_file:
        json.dump(reduced_data, output_file, indent=4)

    print(f"Reduced BBs saved to {output_json}")

if __name__ == "__main__":
    # image_directory = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/RVL-CDIP_original/test"
    # output_json_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_test_100_paddle_ocr.json"
# 
    # detector = RvlCdipBBoxesDetectorPaddleOCR()
    # save_rvl_cdip_bboxes_to_json(detector, image_directory, output_json_path)
# 
    # print(f"BBoxes saved to {output_json_path}")
    # test_calculate_reduction_amount()

    input_json = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_val_50_paddle_ocr.json"
    output_json = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_val_25_paddle_ocr.json"
    reduce_rvl_cdip_bboxes(input_json, output_json, ReductionStep.FROM_50_TO_25)