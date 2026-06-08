import os, json, re, argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model-dir", required=True, help="Path to saved LoRA adapter")
    p.add_argument("--base-model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-image-side", type=int, default=512)
    return p.parse_args()

def extract_json(text: str) -> dict:
    fallback = {"has_lesion": False, "bboxes_ymin_xmin_ymax_xmax": [], "lesion_count": 0}
    
    # Try extracting from <JSON> tags first
    tag_match = re.search(r"<JSON>\s*([\s\S]*?)\s*</JSON>", text)
    if tag_match:
        try:
            return json.loads(tag_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Fallback: try parsing whole response
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Fallback: find any JSON object
    match = re.search(r"\{[\s\S]{0,2000}\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return fallback

def sanitize_box(box):
    y1, x1, y2, x2 = box
    y1 = max(0.0, min(1.0, y1))
    x1 = max(0.0, min(1.0, x1))
    y2 = max(0.0, min(1.0, y2))
    x2 = max(0.0, min(1.0, x2))
    y1, y2 = sorted([y1, y2])
    x1, x2 = sorted([x1, x2])
    return [y1, x1, y2, x2]

def compute_iou(box_a, box_b):
    ay1, ax1, ay2, ax2 = box_a
    by1, bx1, by2, bx2 = box_b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def calculate_multi_box_iou(pred_boxes, gt_boxes):
    if not pred_boxes and not gt_boxes: return 1.0
    if not pred_boxes or not gt_boxes: return 0.0
    matched_ious = []
    temp_gt = list(gt_boxes)
    for p_box in pred_boxes:
        if not temp_gt: break
        current_ious = [compute_iou(p_box, g_box) for g_box in temp_gt]
        best_idx = np.argmax(current_ious)
        if current_ious[best_idx] > 0:
            matched_ious.append(current_ious[best_idx])
            temp_gt.pop(best_idx)
    return np.mean(matched_ious) if matched_ious else 0.0

def draw_boxes(img_path, pred_boxes, gt_boxes, out_path, iou_score):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)

    # Ground Truth: Green
    for box in gt_boxes:
        y1, x1, y2, x2 = [box[0]*h, box[1]*w, box[2]*h, box[3]*w]
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    # Preds: Red
    for box in pred_boxes:
        y1, x1, y2, x2 = [box[0]*h, box[1]*w, box[2]*h, box[3]*w]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    # Label text
    label = f"IoU: {iou_score:.3f} | Pred count: {len(pred_boxes)}"
    d.text((10, 10), label, fill=(255, 255, 0))
    img.save(out_path)

def main():
    args = parse_args()
    jsonl_path = Path(args.dataset_dir) / f"{args.split}.jsonl"
    records = [json.loads(line) for line in open(jsonl_path)]


    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=True)
    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if exists
    results = []
    completed_keys = set()
    checkpoint_path = out_dir / "results_checkpoint.jsonl"

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                completed_keys.add((r["patient_id"], r["modality"], r["slice_index"]))
        print(f"Resuming from checkpoint: {len(results)} slices already done", flush=True)

    records = [r for r in records if (r["patient_id"], r["modality"], r["slice_index"]) not in completed_keys]
    print(f"Remaining slices to process: {len(records)}", flush=True)

    all_ious, correct_class = [], 0
    print(f"Starting inference on {len(records)} records...")

    for i, r in enumerate(records, 1):
        
        # 1. Calling the function properly
        prompt_text = r["prompt"]
        
        img = Image.open(r["image_path"]).convert("RGB")
        w, h = img.size
        scale = min(args.max_image_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Track input length to isolate new tokens
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            
        # 2. Decode ONLY the assistant's new tokens
        resp = processor.tokenizer.decode(
            gen[0][input_len:],
            skip_special_tokens=True
        )

        # 3. Extraction
        pred = extract_json(resp)
        raw_boxes = pred.get("bboxes_ymin_xmin_ymax_xmax", [])
        pred_boxes = [sanitize_box(box) for box in raw_boxes if len(box) == 4]     
        gt_boxes = r.get("bboxes", []) 

        # Debug print for boxes
        if pred_boxes:
            print(f"Slice {i}: Found {len(pred_boxes)} predicted boxes.")

        is_correct = pred.get("has_lesion") == r.get("has_lesion")
        if is_correct: correct_class += 1

        current_iou = calculate_multi_box_iou(pred_boxes, gt_boxes)
        all_ious.append(current_iou)

        if pred_boxes:  # only save images where model found something
            fname = f"{r['patient_id']}_{r['modality']}_s{r['slice_index']:03d}.png"
            draw_boxes(r["image_path"], pred_boxes, gt_boxes, out_dir / fname, current_iou)

        results.append({**r, "pred_boxes": pred_boxes, "iou": current_iou, "correct_class": is_correct, "raw_resp": resp})

        if i % 100 == 0:
            with open(out_dir / "results_checkpoint.jsonl", "w") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            print(f"Checkpoint saved at slice {i}", flush=True)

        if i % 10 == 0:
            print(f"[{i}/{len(records)}] Acc: {correct_class/i:.2f} | mIoU: {np.mean(all_ious):.4f}")

    print(f"\n=== Final Metrics ===\nAccuracy: {correct_class/len(records):.4f}\nmIoU: {np.mean(all_ious):.4f}")
    with open(out_dir / "results.jsonl", "w") as f:
        for res in results: f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()