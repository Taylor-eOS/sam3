import os
import sys
import numpy as np
import torch
from PIL import Image
from utils import load_model, load_image
from settings import DETECTION_THRESHOLD, MASK_THRESHOLD

INPUT_DIR = "input_images"
OUTPUT_DIR = "parallax_layers"
TEXT_PROMPT = input("Item to segment: ")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

def get_sorted_image_paths(input_dir):
    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}")
        sys.exit(1)
    names = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not names:
        print(f"No images found in {input_dir}")
        sys.exit(1)
    return [os.path.join(input_dir, name) for name in names]

def run_segmentation(model, processor, device, image, text_prompt):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=DETECTION_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return results

def select_best_mask(results):
    scores = results["scores"]
    masks = results["masks"]
    if len(masks) == 0:
        return None, None
    best_index = int(torch.argmax(scores))
    return masks[best_index], float(scores[best_index])

def save_cutout(image, mask, output_path):
    mask_bool = mask.cpu().numpy().astype(bool)
    rgba = np.array(image.convert("RGBA"))
    rgba[..., 3] = np.where(mask_bool, 255, 0).astype(np.uint8)
    cutout = Image.fromarray(rgba, mode="RGBA")
    cutout.save(output_path)

def process_folder(input_dir, output_dir, text_prompt):
    os.makedirs(output_dir, exist_ok=True)
    model, processor, device = load_model()
    image_paths = get_sorted_image_paths(input_dir)
    for index, image_path in enumerate(image_paths):
        image = load_image(image_path)
        results = run_segmentation(model, processor, device, image, text_prompt)
        mask, score = select_best_mask(results)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if mask is None:
            print(f"[{index:03d}] {base_name}: no match for \"{text_prompt}\"")
            continue
        output_path = os.path.join(output_dir, f"{index:03d}_{base_name}.png")
        save_cutout(image, mask, output_path)
        print(f"[{index:03d}] {base_name}: saved (score {score:.2f})")

if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR, TEXT_PROMPT)
