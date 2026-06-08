# MedGemma for Multiple Sclerosis Lesion Analysis

This repository contains an implementation and fine-tuning pipeline for **Google’s MedGemma vision–language foundation model**, adapted for **Multiple Sclerosis (MS) lesion analysis in brain MRI**.

The model is explored as an **alternative to traditional CNN-based segmentation networks**, focusing on structured lesion interpretation and bounding box prediction from MRI slices.

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

## 🧬 Model Description

MedGemma is based on the **Gemma 3 architecture**, extended with a biomedical vision encoder.

In this work, it is fine-tuned using **Low-Rank Adaptation (LoRA)** to adapt the model for MS lesion detection and localization.

### Key idea:
Instead of segmentation masks → the model predicts structured outputs:

```json id="medgemma_output_example"
{
  "lesions": [
    {
      "count": 2,
      "bbox": [x1, y1, x2, y2],
      "location": "periventricular white matter"
    }
  ]
}
