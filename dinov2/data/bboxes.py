import os
import csv
import json
import numpy as np
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

def save_rvl_cdip_bboxes_to_csv(detector: RvlCdipBBoxesDetectorPaddleOCR, image_dir: str, output_json: str):
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

if __name__ == "__main__":
    #image_directory = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/CelebA_original"
    #output_csv_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/list_bbox_celeba_mtcnn.csv"
    #
    #detector = CelebABBoxDetectorMTCNN()
    #save_celeba_bboxes_to_csv(detector, image_directory, output_csv_path)
#
    #print(f"BBoxes saved to {output_csv_path}")

    image_directory = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/RVL-CDIP_original/test"
    output_json_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_test_100_paddle_ocr.json"

    detector = RvlCdipBBoxesDetectorPaddleOCR()
    save_rvl_cdip_bboxes_to_csv(detector, image_directory, output_json_path)

    print(f"BBoxes saved to {output_json_path}")