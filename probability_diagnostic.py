import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from utils import parse_args, load_model, load_image
from settings import DETECTION_THRESHOLD, MASK_THRESHOLD

def run_segmentation(model, processor, device, image, text_prompt):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    original_size = inputs.get("original_sizes").tolist()
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=DETECTION_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=original_size,
    )[0]
    return outputs, results, original_size[0]

def make_output_dir(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    run_dir = base_name
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def resize_probabilities(pred_masks_logits, height, width):
    probabilities = torch.sigmoid(pred_masks_logits)
    probabilities = probabilities.unsqueeze(1)
    probabilities = F.interpolate(probabilities, size=(height, width), mode="bilinear", align_corners=False)
    return probabilities.squeeze(1)

def match_query_for_mask(target_mask_bool, probabilities):
    for query_index in range(probabilities.shape[0]):
        candidate_bool = (probabilities[query_index] > MASK_THRESHOLD).cpu().numpy()
        if np.array_equal(candidate_bool, target_mask_bool):
            return query_index
    return None

def save_probability_maps(outputs, results, original_size, run_dir):
    height, width = int(original_size[0]), int(original_size[1])
    pred_masks_logits = outputs.pred_masks[0]
    probabilities = resize_probabilities(pred_masks_logits, height, width)
    masks = results["masks"]
    scores = results["scores"]
    for index in range(len(masks)):
        target_mask_bool = masks[index].cpu().numpy().astype(bool)
        query_index = match_query_for_mask(target_mask_bool, probabilities)
        score = float(scores[index])
        if query_index is None:
            print(f"Instance {index:02d} (score {score:.2f}): could not match a query, skipping.")
            continue
        prob_map = probabilities[query_index].cpu().numpy()
        prob_image = Image.fromarray((prob_map * 255).astype(np.uint8), mode="L")
        prob_path = os.path.join(run_dir, f"instance_{index:02d}_score{score:.2f}_probmap.png")
        prob_image.save(prob_path)
        print(f"Saved {prob_path} (min={prob_map.min():.3f} max={prob_map.max():.3f})")

def main():
    image_path, text_prompt = parse_args()
    model, processor, device = load_model()
    image = load_image(image_path)
    outputs, results, original_size = run_segmentation(model, processor, device, image, text_prompt)
    masks = results["masks"]
    if len(masks) == 0:
        print(f'No objects matching "{text_prompt}" were found.')
        return
    run_dir = make_output_dir(image_path)
    save_probability_maps(outputs, results, original_size, run_dir)
    print(f'Found {len(masks)} object(s) matching "{text_prompt}"')
    print(f"Probability maps saved to {run_dir}")

if __name__ == "__main__":
    main()
