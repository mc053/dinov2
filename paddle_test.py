from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/RVL-CDIP/RVL-CDIP_original/val/image_0_label_15.jpg"
result = ocr.ocr(img_path,rec=False)
print(result[0])