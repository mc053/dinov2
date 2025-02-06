import csv
import os
import json
import cv2
import numpy as np
from enum import Enum
from PIL import Image
from tqdm import tqdm

# Adjust main method before starting job.

class CelebAAnonymizer:
    def anonymize_celeba_imgs(self, input_path: str, output_path: str, bbox_csv_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        bbox_data = self._load_bboxes(bbox_csv_path)
        images = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png'))][:100] # for testing with first 100 images.

        for image_name in tqdm(images, desc="Anonymizing images"):
            input_image_path = os.path.join(input_path, image_name)
            output_image_path = os.path.join(output_path, image_name)

            try:
                image = Image.open(input_image_path).convert("RGB")
                mask = self._generate_mask(image.size, bbox_data.get(image_name))
                anonymized_image = self._apply_anonymization_for_mask(image, mask)
                anonymized_image.save(output_image_path)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    def _load_bboxes(self, bbox_csv_path: str) -> dict:
        bbox_data = {}
        with open(bbox_csv_path, "r") as bbox_csv:
            csvreader = csv.DictReader(bbox_csv)
            for row in csvreader:
                try:
                    bbox_data[row["image_id"]] = (
                        int(row["x_1"]),
                        int(row["y_1"]),
                        int(row["width"]),
                        int(row["height"])
                    )
                except Exception as e:
                    print(f"Error processing row: {row} - {e}")
                    continue

        return bbox_data

    def _generate_mask(self, image_size: tuple, bbox: tuple) -> np.ndarray:
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        if not bbox:
            raise ValueError("No bounding box found for this image. This should not happen after sorting out unrecognized images.")
        
        x_1, y_1, width, height = bbox
        x_2, y_2 = x_1 + width, y_1 + height
        mask[y_1:y_2, x_1:x_2] = 1

        return np.stack([mask] * 3, axis=-1)  # RGB

    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        raise NotImplementedError

class CelebAAnonymizerMaskOut(CelebAAnonymizer):
    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        img_array = np.array(image)
        anonymized_array = img_array * (1 - mask)

        return Image.fromarray(anonymized_array.astype(np.uint8))

class CelebAAnonymizerPixelation(CelebAAnonymizer):
    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray, pixel_ratio=0.1) -> Image.Image:
        width, height = image.size
    
        new_width = int(width * pixel_ratio)
        new_height = int(height * pixel_ratio)
        
        pixelated_image = image.resize((new_width, new_height), resample=Image.NEAREST)
        pixelated_image = pixelated_image.resize((width, height), resample=Image.NEAREST)
        
        original_array = np.array(image)
        pixelated_array = np.array(pixelated_image)
        
        anonymized_array = original_array * (1 - mask) + pixelated_array * mask
        
        return Image.fromarray(anonymized_array.astype(np.uint8))

# https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1
class CelebAAnonymizerGaussianBlur(CelebAAnonymizer):
    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray, sigma=8):
        img_array = np.array(image)
        blurred_image = cv2.GaussianBlur(img_array, (0, 0), sigmaX=sigma)
        anonymized_array = img_array * (1 - mask) + blurred_image * mask

        return Image.fromarray(anonymized_array.astype(np.uint8))

class RvlCdipAnonymizer:
    def anonymize_rvlcdip_imgs(self, input_path: str, output_path: str, bbox_json_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        bbox_data = self._load_bboxes(bbox_json_path)
        images = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png'))]    # [:100] for testing with first 100 images.

        for image_name in tqdm(images, desc="Anonymizing images"):
            input_image_path = os.path.join(input_path, image_name)
            output_image_path = os.path.join(output_path, image_name)

            if image_name not in bbox_data:
                raise ValueError(f"No bounding box found for {image_name}. This should not happen after sorting out unrecognized images.")

            try:
                image = Image.open(input_image_path).convert("RGB")
                mask = self._generate_mask(image.size, bbox_data[image_name]["bbs"])
                anonymized_image = self._apply_anonymization_for_mask(image, mask)
                anonymized_image.save(output_image_path)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    def _load_bboxes(self, bbox_json_path: str) -> dict:
        with open(bbox_json_path, "r") as f:
            return json.load(f)

    def _generate_mask(self, image_size: tuple, bboxes: list) -> np.ndarray:
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        for bbox in bboxes:
            # See https://github.com/Mushroomcat9998/PaddleOCR/blob/main/tools/infer/utility.py#L208
            reshaped_bbox = np.reshape(bbox, (-1, 2)).astype(np.int64)
            x_min, y_min = reshaped_bbox.min(axis=0)
            x_max, y_max = reshaped_bbox.max(axis=0)
            mask[y_min:y_max, x_min:x_max] = 1

        return np.stack([mask] * 3, axis=-1)  # RGB

    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        raise NotImplementedError

class RVLCDIPAnonymizerMaskOut(RvlCdipAnonymizer):
    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        img_array = np.array(image)
        anonymized_array = img_array * (1 - mask)

        return Image.fromarray(anonymized_array.astype(np.uint8))

class RVLCDIPAnonymizerPixelation(RvlCdipAnonymizer):
    def _apply_anonymization_for_mask(self, image: Image.Image, mask: np.ndarray, pixel_ratio=0.1) -> Image.Image:
        width, height = image.size
    
        new_width = int(width * pixel_ratio)
        new_height = int(height * pixel_ratio)
        
        pixelated_image = image.resize((new_width, new_height), resample=Image.NEAREST)
        pixelated_image = pixelated_image.resize((width, height), resample=Image.NEAREST)
        
        original_array = np.array(image)
        pixelated_array = np.array(pixelated_image)
        
        anonymized_array = original_array * (1 - mask) + pixelated_array * mask
        
        return Image.fromarray(anonymized_array.astype(np.uint8))

if __name__ == "__main__":
    input_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/CelebA_original/val"
    output_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/CelebA_blurred/val"
    bbox_csv_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/list_bbox_celeba_mtcnn.csv"
# 
    anonymizer = CelebAAnonymizerGaussianBlur()
# 
    print("Starting anonymization...")
    anonymizer.anonymize_celeba_imgs(input_path, output_path, bbox_csv_path)
    print(f"Anonymization completed. Anonymized images saved in {output_path}")

    # input_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/RVL-CDIP_original/train"
    # output_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/RVL-CDIP_100_pixelated/train"
    # bbox_json_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_train_100_paddle_ocr.json"
# 
    # anonymizer = RVLCDIPAnonymizerPixelation()
# 
    # print("Starting anonymization...")
    # anonymizer.anonymize_rvlcdip_imgs(input_path, output_path, bbox_json_path)
    # print(f"Anonymization completed. Anonymized images saved in {output_path}")

    # '/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_train_100_paddle_ocr.json'
    # /home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/list_bboxes_rvl_cdip_train_100_paddle_ocr.csv