import os
import json
import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, TaskType



# ---------------------------------------------
# TARGET
# ---------------------------------------------
def make_target(record):
    return json.dumps({
        "has_lesion": record.get("has_lesion", False),
        "lesion_count": len(record.get("bboxes", [])),
        "bboxes_ymin_xmin_ymax_xmax": record.get("bboxes", [])
    })


# ---------------------------------------------
# DATASET
# ---------------------------------------------
class MSBBoxDataset(Dataset):
    def __init__(self, jsonl_path, max_image_side=512):
        self.records = []
        self.max_image_side = max_image_side

        with open(jsonl_path, "r") as f:
            for line in f:
                r = json.loads(line)
                if os.path.exists(r["image_path"]):
                    self.records.append(r)

        print(f"Loaded {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        img = Image.open(r["image_path"]).convert("RGB")

        w, h = img.size
        scale = min(self.max_image_side / max(w, h), 1.0)

        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))

        return {
            "image": img,
            "prompt": r["prompt"],
            "target": make_target(r)
        }


# ---------------------------------------------
# COLLATE (FIXED)
# ---------------------------------------------
# def make_collate_fn(processor):
#     def collate(batch):
#         conversations = []
 
#         for item in batch:
#             conversation = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": item["image"]},
#                         {"type": "text", "text": item["prompt"]},
#                     ],
#                 },
#                 {
#                     "role": "assistant",
#                     "content": [
#                         {"type": "text", "text": item["target"]},
#                     ],
#                 },
#             ]
#             conversations.append(conversation)
 
#         encoded = processor.apply_chat_template(
#             conversations,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#             padding=True,
#             add_generation_prompt=False,
#         )
 
#         # Start with all tokens masked
#         labels = torch.full_like(encoded["input_ids"], -100)
 
#         for i, conversation in enumerate(conversations):
#             # Encode ONLY the user/prompt part to find where it ends
#             prompt_only = processor.apply_chat_template(
#                 [conversation[0]],  # only the user turn
#                 tokenize=True,
#                 return_dict=True,
#                 return_tensors="pt",
#                 add_generation_prompt=True,  # include the assistant turn opener
#             )
#             prompt_len = prompt_only["input_ids"].shape[1]
 
#             # Only unmask the assistant response tokens
#             labels[i, prompt_len:] = encoded["input_ids"][i, prompt_len:]
 
#         # Re-mask padding
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         encoded["labels"] = labels
 
#         return encoded
 
#     return collate  # < returns the inner function

def make_collate_fn(processor):
    def collate(batch):
        item = batch[0]  # batch_size=1, so just take first item

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["target"]},
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = processor(
            text=[text],
            images=[item["image"]],
            return_tensors="pt",
            padding=False,
        )

        # Label masking
        labels = encoded["input_ids"].clone()

        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item["prompt"]}
            ]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = processor.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels[0, :prompt_len] = -100
        labels[labels == processor.tokenizer.pad_token_id] = -100

        encoded["labels"] = labels
        return encoded

    return collate
# ---------------------------------------------
# TRAIN
# ---------------------------------------------

def train(args):

    device = "cuda"

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    processor = AutoProcessor.from_pretrained(args.model_id)

    # model (IMPORTANT: NO model.to("cuda"))
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # data
    train_ds = MSBBoxDataset(os.path.join(args.dataset_dir, "train.jsonl"))
    val_ds = MSBBoxDataset(os.path.join(args.dataset_dir, "val.jsonl"))

    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TRAIN LOOP
    for epoch in range(args.epochs):

        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):

            #  move EVERYTHING to cuda safely
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss
            if loss is None:
                continue

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 20 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                if outputs.loss is not None:
                    val_loss += outputs.loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(f"\nEpoch {epoch}: train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            model.save_pretrained(out_dir / "best_model")
            processor.save_pretrained(out_dir / "best_model")
            print("Saved best model")

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()
    train(args)