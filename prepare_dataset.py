"""
prepare_dataset.py

Converts NIfTI volumes + segmentation masks into a fine-tuning dataset.
Outputs:
  - PNG slices for each patient/modality
  - train.jsonl, val.jsonl and test.jsonl with (image_path, bbox, label) per slice

Usage:
  python3 prepare_dataset.py \
    --data-root /mnt/jana.ristevska/MS_Project/open_ms_data \
    --out-dir /mnt/jana.ristevska/MS_Project/dataset_out \
    --png-size 512 \
    --slice-axis 2 \
    --trim-frac 0.15
"""

import os, json, re, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.measure import label, regionprops

MODALITIES = ["FLAIR", "T1"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--png-size", type=int, default=512)
    p.add_argument("--slice-axis", type=int, default=2, choices=[0,1,2])
    p.add_argument("--trim-frac", type=float, default=0.15,
               help="Trim empty slices from beginning/end")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def norm_to_uint8(vol, p_low=1, p_high=99):
    vmin, vmax = np.percentile(vol, (p_low, p_high))
    vol = np.clip(vol, vmin, vmax).astype(np.float32)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return (vol * 255).astype(np.uint8)

def get_slice(vol, axis, k):
    if axis == 2: return vol[:, :, k]
    if axis == 1: return vol[:, k, :]
    return vol[k, :, :]

def mask_to_bboxes(mask_slice, min_area=200):
    """
    Finds individual MS lesions and returns coordinates in 
    Normalized bounding boxes in [0,1]    
    Extract lesion bounding boxes while filtering tiny noisy lesions.

    Args:
        mask_slice: binary lesion mask
        min_area: minimum lesion area in pixels
    """
    bboxes = []
    # label(mask_slice) identifies separate "blobs" of lesions
    labeled = label(mask_slice) 
    h, w = mask_slice.shape
    
    for region in regionprops(labeled):
        # FILTER VERY SMALL LESIONS
        if region.area < min_area:
            continue
       
        # region.bbox returns (min_row, min_col, max_row, max_col)
        # which corresponds exactly to (ymin, xmin, ymax, xmax)
        ymin, xmin, ymax, xmax = region.bbox
        
        #Normalize to [0,1] scale
        norm_box = [
            round(ymin / h, 4),
            round(xmin / w, 4),
            round(ymax / h, 4),
            round(xmax / w, 4)
        ]
        bboxes.append(norm_box)
    return bboxes

def stratified_split(pids, val_frac, seed): # Added pids and val_frac
    rng = np.random.default_rng(seed)
    pids_list = list(pids) # Use the passed list
    rng.shuffle(pids_list)

    n_val = max(1, int(len(pids_list) * val_frac))
    val_pids = pids_list[:n_val]
    train_pids = pids_list[n_val:]
    
    return train_pids, val_pids

def make_prompt(modality: str) -> str:
    seq_map = {
        "flair": "FLAIR",
        "t1": "T1"
    }

    seq = seq_map.get(str(modality).lower(), "FLAIR")

    return f"""You are an expert neuroradiologist analyzing a brain MRI slice ({seq} sequence).

Task:
- Detect ALL Multiple Sclerosis lesions visible in the white matter
- Return bounding boxes in normalized coordinates [0,1]

Return ONLY valid JSON inside <JSON> tags, no other text.

Example output with lesions:
<JSON>
{{"has_lesion": true, "lesion_count": 2, "bboxes_ymin_xmin_ymax_xmax": [[0.31, 0.44, 0.38, 0.52], [0.61, 0.21, 0.67, 0.29]]}}
</JSON>

Example output without lesions:
<JSON>
{{"has_lesion": false, "lesion_count": 0, "bboxes_ymin_xmin_ymax_xmax": []}}
</JSON>

Now analyze the image and return your answer:
"""
def process_patient(pid, patient_path, out_dir, png_size, slice_axis, trim_frac, inference_mode=False):
    """
    Args:
        patient_path: Path object pointing to MS_Project/open_ms_data/train/patient01
    """
    records = []
    vols = {}
    
    # Mapping based on your specific file structure
    modality_files = {
        "FLAIR": patient_path / "FLAIR.nii.gz",
        "T1": patient_path / "T1.nii.gz"
    }
    mask_path = patient_path / "lesion.nii.gz"

    # 1. Load Modalities
    for mod, path in modality_files.items():
        if path.exists():
            vols[mod] = nib.load(str(path)).get_fdata().astype(np.float32)
        else:
            print(f"  WARNING: {mod} not found for {pid}")

    if "FLAIR" not in vols:
        return records

    # 2. Load Mask
    if not mask_path.exists():
        if inference_mode:
            mask_vol = np.zeros_like(vols["FLAIR"], dtype=np.uint8)  # all-zero mask
        else:
            print(f"  WARNING: No lesion mask for {pid}")
            return records
    else:
        mask_vol = (nib.load(str(mask_path)).get_fdata() > 0).astype(np.uint8)

    # 3. Slicing Setup
    n_total = vols["FLAIR"].shape[slice_axis]
    lo, hi = int(n_total * trim_frac), int(n_total * (1.0 - trim_frac))
    
    # 4. Export and Labeling
    pid_dir = Path(out_dir) / "images" / pid
    pid_dir.mkdir(parents=True, exist_ok=True)

    for mod, vol in vols.items():
        vol_u8 = norm_to_uint8(vol)
        
        for k in range(lo, hi):
            # Extract slices
            img_sl = get_slice(vol_u8, slice_axis, k)
            mask_sl = get_slice(mask_vol, slice_axis, k)

            # -- ADD THIS --
            # Skip empty slices (all-zero = outside brain volume)
            if img_sl.max() == 0:
                continue
            # -------------

            # Resize if necessary
            if png_size:
                img_pil = Image.fromarray(img_sl).resize((png_size, png_size), Image.BILINEAR)
                m_pil = Image.fromarray(mask_sl * 255).resize((png_size, png_size), Image.NEAREST)
                mask_sl_proc = np.array(m_pil) > 0
            else:
                img_pil = Image.fromarray(img_sl)
                mask_sl_proc = mask_sl > 0

            # Save Image
            fname = f"{pid}_{mod}_s{k:03d}.png"
            img_save_path = str(pid_dir / fname)
            img_pil.convert("L").save(img_save_path)

            # Extract Bboxes
            boxes = mask_to_bboxes(mask_sl_proc)

            records.append({
                "patient_id": pid,
                "modality": mod,
                "slice_index": k,
                "image_path": img_save_path,
                "has_lesion": len(boxes) > 0,
                "bboxes": boxes,  # normalized [ymin, xmin, ymax, xmax] in [0,1]
                "prompt": make_prompt(mod)
            })

    return records

def main():
    args = parse_args()

    # Path setup based on your MS_Project/open_ms_data/train structure
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "inference"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all patient folders in the train directory
    # This replaces the need for dataset.json
    train_patient_paths = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    train_patient_ids = [p.name for p in train_patient_paths]
    
    if not train_patient_ids:
        print(f"Error: No patient folders found in {train_dir}")
        return

    print(f"Total train patients discovered: {len(train_patient_ids)}")

    # 2. Stratified Split 
    # (Using a simple 80/20 split or your stratified_split function)
    # Since MS doesn't have "tumor types" in the same way, we split by ID
    train_pids_list, val_pids_list = stratified_split(
        train_patient_ids,
        val_frac=0.2, 
        seed=args.seed
        )
    train_pids = set(train_pids_list)
    val_pids = set(val_pids_list)

    print(f"Split: {len(train_pids)} train, {len(val_pids)} val")

    # --------------------------------------------------
    # TEST PATIENTS
    # --------------------------------------------------

    test_patient_ids = []

    if test_dir.exists():
        test_patient_paths = sorted([p for p in test_dir.iterdir() if p.is_dir()])
        test_patient_ids = [p.name for p in test_patient_paths]

        print(f"Total TEST patients discovered: {len(test_patient_ids)}")
    else:
        print(f"WARNING: Test directory not found: {test_dir}")

    # Save split info for reproducibility
    split_info = {
        "train": sorted(list(train_pids)), 
        "val": sorted(list(val_pids)),
        "test": sorted(list(test_patient_ids))

    }
    (out_dir / "split.json").write_text(
        json.dumps(split_info, indent=2)
    )

    # 3. Process Patients
    train_records = []
    val_records = []
    test_records = []

    for i, pid in enumerate(train_patient_ids, 1):
        split = "val" if pid in val_pids else "train"
        patient_path = train_dir / pid
        
        print(f"\n[{i}/{len(train_patient_ids)}] {pid} ({split})")
        
        # Calling your updated process_patient
        recs = process_patient(
            pid=pid,
            patient_path=patient_path,
            out_dir=out_dir,
            png_size=args.png_size,
            slice_axis=args.slice_axis,
            trim_frac=args.trim_frac,
        )

        n_lesion = sum(1 for r in recs if r.get('has_lesion'))
        print(f"  -> {len(recs)} slices ({n_lesion} with MS lesions)")

        if split == "train":
            train_records.extend(recs)
        else:
            val_records.extend(recs)

    # --------------------------------------------------
    # PROCESS TEST
    # --------------------------------------------------

    if test_patient_ids:

        print("\nProcessing TEST patients...")

        for i, pid in enumerate(test_patient_ids, 1):

            patient_path = test_dir / pid

            print(f"\n[TEST {i}/{len(test_patient_ids)}] {pid}")

            recs = process_patient(
                pid=pid,
                patient_path=patient_path,
                out_dir=out_dir,
                png_size=args.png_size,
                slice_axis=args.slice_axis,
                trim_frac=args.trim_frac,
                inference_mode=True

            )

            n_lesion = sum(1 for r in recs if r.get("has_lesion"))

            print(f"  -> {len(recs)} slices ({n_lesion} with MS lesions)")

            test_records.extend(recs)

    # --------------------------------------------------
    # WRITE JSONL FILES
    # --------------------------------------------------

    split_datasets = [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]

    # 4. Write JSONL files for MedGemma training
    for split_name, records in split_datasets:

        if not records:
            continue

        path = out_dir / f"{split_name}.jsonl"

        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        n_pos = sum(1 for r in records if r.get("has_lesion"))

        pct = (n_pos / len(records) * 100) if records else 0

        print(
            f"\nSaved {split_name}.jsonl: "
            f"{len(records)} records, "
            f"{n_pos} positive ({pct:.1f}%)"
        )

    print(f"\nDone! Dataset ready at: {out_dir}")

if __name__ == "__main__":
    main()
