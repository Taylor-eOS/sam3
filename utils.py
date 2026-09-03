import os
import sys
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from settings import MODEL_NAME

def parse_args():
    image_path = input("Path to input image: ").strip() or "tank.jpeg"
    if not image_path:
        print("No image path given.")
        sys.exit(1)
    text_prompt = input('Object to segment (e.g. "tank"): ').strip() or "tank"
    if not text_prompt:
        print("No text prompt given.")
        sys.exit(1)
    return image_path, text_prompt

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN environment variable is not set.")
        sys.exit(1)
    model = Sam3Model.from_pretrained(MODEL_NAME, device_map=device, token=token)
    processor = Sam3Processor.from_pretrained(MODEL_NAME, token=token)
    return model, processor, device

def load_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Could not find image {image_path}")
        sys.exit(1)
    return Image.open(image_path).convert("RGB")
