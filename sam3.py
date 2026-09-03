import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
from utils import parse_args, load_model, load_image
from settings import DETECTION_THRESHOLD, MASK_THRESHOLD

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

def make_output_dir(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    run_dir = base_name #os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_instance_cutout(image, mask, score, index, run_dir):
    mask_bool = mask.cpu().numpy().astype(bool)
    rgba = np.array(image.convert("RGBA"))
    rgba[..., 3] = np.where(mask_bool, 255, 0).astype(np.uint8)
    cutout = Image.fromarray(rgba, mode="RGBA")
    cutout_path = os.path.join(run_dir, f"instance_{index:02d}_score{score:.2f}.png")
    cutout.save(cutout_path)
    mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
    mask_path = os.path.join(run_dir, f"instance_{index:02d}_mask.png")
    mask_img.save(mask_path)

def draw_preview(image, boxes, scores, run_dir):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for index, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = f"{index:02d} {score:.2f}"
        draw.text((x1 + 4, y1 + 4), label, fill=(255, 0, 0), font=font)
    preview_path = os.path.join(run_dir, "preview.png")
    preview.save(preview_path)

def main():
    image_path, text_prompt = parse_args()
    model, processor, device = load_model()
    image = load_image(image_path)
    results = run_segmentation(model, processor, device, image, text_prompt)
    masks = results["masks"]
    boxes = results["boxes"]
    scores = results["scores"]
    if len(masks) == 0:
        print(f'No objects matching "{text_prompt}" were found.')
        return
    run_dir = make_output_dir(image_path)
    for index in range(len(masks)):
        save_instance_cutout(image, masks[index], float(scores[index]), index, run_dir)
    draw_preview(image, boxes, scores, run_dir)
    print(f'Found {len(masks)} object(s) matching "{text_prompt}"')
    print(f"Output saved to {run_dir}")

if __name__ == "__main__":
    main()
