import os
import csv
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from typing import Optional, Tuple
from tqdm import tqdm

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

def save_bboxes_to_csv(detector: CelebABBoxDetectorMTCNN, image_dir: str, output_csv: str, max_images: int = 100):
    images = sorted(os.listdir(image_dir))[:max_images]
    output_path = os.path.abspath(output_csv)

    with open(output_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_id", "x_1", "y_1", "width", "height"])

        for image in tqdm(images, desc="Processing images"):
            img_path = os.path.join(image_dir, image)
            bbox = detector.detect_celeba_bbox(img_path)

            if bbox:
                csvwriter.writerow([image, *bbox])

if __name__ == "__main__":
    image_directory = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/CelebA_original"
    output_csv_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/list_bbox_celeba_mtcnn.csv"
    
    detector = CelebABBoxDetectorMTCNN()
    save_bboxes_to_csv(detector, image_directory, output_csv_path, max_images=100)

    print(f"BBoxes saved to {output_csv_path}")