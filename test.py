import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR

# (이미 설치된 대로)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
# ocr = PaddleOCR(lang='korean', use_textline_orientation=True)
ocr = PaddleOCR(lang='korean', use_textline_orientation=True, use_angle_cls=True)

img_path = "ASRC상장.jpg"  # 실제로 테스트할 이미지 파일 경로
pil = Image.open(img_path).convert("RGB")

# 1) BLIP 캡션
inputs = processor(images=pil, return_tensors="pt").to(device)
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("BLIP caption:", caption)

# 2) PaddleOCR 텍스트 추출
result = ocr.ocr(img_path, cls=True)
print("PaddleOCR result:", result)
