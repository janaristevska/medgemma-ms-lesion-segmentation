import os, json
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score
)

def bbox_to_mask(bboxes, h=512, w=512):
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in bboxes:
        ymin, xmin, ymax, xmax = box
        ymin = max(0.0, min(1.0, ymin))
        xmin = max(0.0, min(1.0, xmin))
        ymax = max(0.0, min(1.0, ymax))
        xmax = max(0.0, min(1.0, xmax))
        ymin, ymax = sorted([ymin, ymax])
        xmin, xmax = sorted([xmin, xmax])
        y1, x1 = int(ymin * h), int(xmin * w)
        y2, x2 = int(ymax * h), int(xmax * w)
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = 1
    return mask

def compute_dice(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    return (2 * intersection / total) if total > 0 else 1.0

def compute_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return (intersection / union) if union > 0 else 1.0

def main():
    results_path = "/home/hpc/users/ml_models/jana.ristevska/MedGemma/inference_output/results.jsonl"
    evaluate_dir = Path("/home/hpc/users/ml_models/jana.ristevska/MS_Project/open_ms_data/evaluate")

    records = [json.loads(l) for l in open(results_path)]

    by_patient = {}
    for r in records:
        pid = r["patient_id"]
        if pid not in by_patient:
            by_patient[pid] = []
        by_patient[pid].append(r)

    all_dice, all_iou = [], []
    all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

    for pid, slices in by_patient.items():
        mask_path = evaluate_dir / pid / "lesion.nii.gz"
        if not mask_path.exists():
            print(f"WARNING: No ground truth for {pid}, skipping")
            continue

        mask_vol = (nib.load(str(mask_path)).get_fdata() > 0).astype(np.uint8)
        print(f"Patient {pid}: mask shape {mask_vol.shape}", flush=True)

        total_slices = 0
        for r in slices:
            if r["modality"] != "FLAIR":
                continue

            k = r["slice_index"]
            if k >= mask_vol.shape[2]:
                continue

            gt_slice = mask_vol[:, :, k]

            from PIL import Image as PILImage
            gt_pil = PILImage.fromarray(gt_slice * 255).resize(
                (512, 512), PILImage.NEAREST
            )
            gt_mask = (np.array(gt_pil) > 0).astype(np.uint8)
            pred_mask = bbox_to_mask(r.get("pred_boxes", []))

            dice = compute_dice(pred_mask, gt_mask)
            iou = compute_iou(pred_mask, gt_mask)

            all_dice.append(dice)
            all_iou.append(iou)

            # accumulate confusion matrix counts instead of storing all pixels
            all_tp += int(((pred_mask == 1) & (gt_mask == 1)).sum())
            all_fp += int(((pred_mask == 1) & (gt_mask == 0)).sum())
            all_fn += int(((pred_mask == 0) & (gt_mask == 1)).sum())
            all_tn += int(((pred_mask == 0) & (gt_mask == 0)).sum())

            total_slices += 1
            if total_slices % 100 == 0:
                print(f"  {pid}: processed {total_slices} slices so far...", flush=True)

        print(f"  {pid}: done ({total_slices} slices)", flush=True)

    print(f"\nTotal slices evaluated: {len(all_dice)}", flush=True)

    if not all_dice:
        print("ERROR: No slices were evaluated.")
        return

    # compute metrics from confusion matrix counts
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = all_tn / (all_tn + all_fp) if (all_tn + all_fp) > 0 else 0.0
    bal_acc = (recall + specificity) / 2

    total = all_tp + all_fp + all_fn + all_tn
    tp_rate = all_tp / total
    tn_rate = all_tn / total
    macro_f1 = (f1 + (2 * specificity * (1 - recall) / (specificity + (1 - recall)) if (specificity + (1 - recall)) > 0 else 0.0)) / 2

    print(f"\n=== Evaluation Metrics ===")
    print(f"Dice Score:          {np.mean(all_dice):.4f}")
    print(f"IoU:                 {np.mean(all_iou):.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Balanced Accuracy:   {bal_acc:.4f}")

if __name__ == "__main__":
    main()
