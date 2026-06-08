# MedGemma for Multiple Sclerosis Lesion Analysis

This repository contains an implementation and fine-tuning pipeline for **Google's MedGemma vision–language foundation model**, adapted for **Multiple Sclerosis (MS) lesion analysis in brain MRI**.

The model is explored as an alternative to traditional CNN-based segmentation networks, focusing on structured lesion interpretation and bounding box prediction from MRI slices.

---

## 🧠 Overview

Multiple Sclerosis lesion segmentation is typically addressed using dense prediction models such as 3D CNNs and U-Nets. In contrast, this project investigates the use of a **vision–language foundation model (MedGemma)** for lesion analysis.

MedGemma combines:

- Vision encoding for medical images
- Natural language reasoning via a large language model
- Structured text generation capabilities

Instead of producing voxel-level segmentation masks, MedGemma generates:

- Lesion presence descriptions
- Lesion counts
- Spatial localization (bounding boxes in JSON format)

---

## 🖼️ Framework Overview

The figure below illustrates the MedGemma-based vision–language framework for MS lesion analysis. MRI slices are paired with task-specific prompts and processed by a LoRA fine-tuned MedGemma model to produce structured lesion predictions in JSON format, including bounding boxes and lesion attributes.

![MedGemma Framework](figures/medgemma_overview.png)

---

## 🧬 Model Description

MedGemma is based on the **Gemma 3 architecture**, extended with a biomedical vision encoder.
In this work, it is fine-tuned using **Low-Rank Adaptation (LoRA)** to adapt the model for MS lesion detection and localization.

Instead of segmentation masks, the model predicts structured outputs:

```json
{
  "lesions": [
    {
      "count": 2,
      "bbox": [x1, y1, x2, y2],
      "location": "periventricular white matter"
    }
  ]
}
```

---

## 🖼️ Input Format

The model is trained on 2D MRI slices extracted from 3D volumes.
Each training sample consists of:

- MRI slice image (FLAIR or T1-weighted)
- Instruction prompt (template-based)
- Ground-truth bounding boxes derived from segmentation masks

Example prompt:

```
Analyze the MRI slice and identify any MS lesions.
Return bounding boxes in JSON format with lesion location.
```

---

## 📦 Dataset

MedGemma is trained on a subset of 20 MS patients from publicly available datasets.

- MRI modalities: FLAIR, T1-weighted
- Preprocessed data: co-registered, bias-corrected
- Labels: segmentation masks converted to bounding boxes
- Format: JSONL (image–text pairs)

---

## 🏋️ Training

The model is fine-tuned using:

- Low-Rank Adaptation (LoRA)
- Parameter-efficient training strategy
- Instruction tuning on MRI slice–prompt pairs

Key characteristics:

- 2D slice-based training (not volumetric)
- Supervised learning using bounding box targets
- Structured JSON output supervision
- Small dataset (~20 patients)

---

## 📊 Evaluation

The model is evaluated using both quantitative and qualitative metrics.

Quantitative metrics:

- Dice Similarity Coefficient (after bbox → mask conversion)
- Precision
- Recall
- Balanced Accuracy
- mIoU (mean Intersection over Union)

Qualitative evaluation:

- Ability to generate coherent lesion descriptions
- Spatial reasoning in medical context
- Consistency of structured JSON outputs

---

## 📈 Results

| Model            | Dice   | Precision | Recall | Balanced Accuracy |
|------------------|--------|-----------|--------|-------------------|
| MedGemma (LoRA)  | 0.438  | 0.042     | 0.064  | 0.530             |
| 3D CNN           | 0.374  | 0.261     | 0.912  | —                 |
| nnU-Net          | higher | higher    | higher | higher            |

Key observations:

- MedGemma achieves reasonable Dice after bounding box conversion
- Very low precision and recall due to bounding box approximation
- Performance limited by small dataset and task mismatch

---

## ⚠️ Limitations

- Outputs are bounding boxes, not pixel-level segmentation
- Conversion from bounding boxes to masks introduces false positives
- Very limited dataset (~20 patients)
- Not optimized for dense segmentation tasks
- Sensitive to individual patient variability

---

## 🔬 Discussion

Unlike CNN-based models, MedGemma operates at a semantic reasoning level, making it suitable for:

- Lesion interpretation
- Clinical explanation generation
- Structured reporting

However, for precise segmentation tasks, models such as:

- 3D U-Net
- nnU-Net

remain superior under current dataset conditions.

MedGemma should be considered an exploratory foundation model approach, not a replacement for segmentation architectures.

---

## 🚀 Future Work

- Improve instruction tuning for medical localization tasks
- Increase dataset size for robustness
- Explore hybrid systems (MedGemma + U-Net)
- Replace bounding box outputs with segmentation-aware decoding
- Extend to multi-slice or volumetric reasoning

---

## 📁 Repository Structure

```
.
├── data/
├── prompts/
├── models/
├── training/
├── inference/
├── utils/
├── figures/
│   └── medgemma_overview.png
└── README.md
```

---

## 📌 Citation

If you use this work, please cite:

- MedGemma / Gemma 3 (Google)
- Lesjak et al. MS MRI datasets

---

## 📬 Contact

Your Name
Your Email / GitHub
