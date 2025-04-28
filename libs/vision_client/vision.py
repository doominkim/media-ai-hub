# pip install transformers torch pillow 필요
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# 모델/프로세서 전역 캐싱
_model = None
_processor = None

def describe_image(image_path: str) -> str:
    global _model, _processor
    if _model is None or _processor is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    image = Image.open(image_path)
    # 예시: 간단한 프롬프트로 분류
    candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a person", "a photo of a car"]
    inputs = _processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
    return candidate_labels[best_idx]

def analyze_emotion(image_path: str) -> str:
    # 실제 vision 모델 연동 전용 stub
    return f"dummy emotion for {image_path}" 