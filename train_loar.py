"""
LoRA fine-tuning of HuggingFaceTB/SmolVLM-500M-Instruct for
Construction Site Safety Monitoring.

Categories:
1. Working at Height
2. Site Entry / PPE Compliance
3. Slips / Trips / Falls

Designed for:
- Transformers 5.16.1
- PEFT
- NVIDIA GTX 1650 4GB
- Windows
- Small raw HF Dataset
"""

import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

DATA_JSON = "dataset/train.json"
IMAGES_DIR = "dataset/images"

OUTPUT_DIR = "smolvlm-safety-lora"

random.seed(42)


SYSTEM_PROMPT = (
    "You are a construction site safety inspector. "
    "Analyze the image and classify the situation as SAFE, WARNING, "
    "or DANGER, focusing on Working at Height, Site Entry/PPE "
    "Compliance, and Slips/Trips/Falls. "
    "Give a short explanation citing visual evidence."
)


# ============================================================
# 1. LOAD RAW DATASET
# ============================================================

def load_raw_dataset(json_path: str) -> Dataset:

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    cleaned = []

    for r in records:

        image_path = r.get("image") or r.get("image_path")
        question = r.get("question") or r.get("prompt")
        answer = r.get("answer") or r.get("response")

        if not image_path or not question or not answer:
            continue

        # Resolve image path
        if os.path.isabs(image_path):

            full_path = image_path

        elif image_path.startswith(IMAGES_DIR):

            full_path = image_path

        else:

            full_path = os.path.join(
                IMAGES_DIR,
                os.path.basename(image_path)
            )

        # Make sure image exists
        if not os.path.exists(full_path):

            print(f"Skipping missing image: {full_path}")
            continue

        cleaned.append(
            {
                "image_path": full_path,
                "question": question.strip(),
                "answer": answer.strip(),
            }
        )

    if not cleaned:

        raise ValueError(
            f"No valid records found in {json_path}."
        )

    print(f"Loaded {len(cleaned)} raw training examples.")

    return Dataset.from_list(cleaned)


# ============================================================
# 2. SMOLVLM COLLATOR
# ============================================================

class SafetyCollator:

    def __init__(self, processor):

        self.processor = processor

        # <image> token ID
        try:

            self.image_token_id = (
                processor.tokenizer.convert_tokens_to_ids("<image>")
            )

        except Exception:

            self.image_token_id = None

        # Actual tokenizer padding ID
        self.pad_token_id = processor.tokenizer.pad_token_id


    def __call__(self, examples):

        # ----------------------------------------------------
        # GTX 1650 -> batch size MUST remain 1
        # ----------------------------------------------------

        if len(examples) != 1:

            raise RuntimeError(
                f"SafetyCollator expects batch size 1, "
                f"but received {len(examples)}."
            )

        ex = examples[0]

        image_path = ex["image_path"]

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )


        # ----------------------------------------------------
        # IMPORTANT
        #
        # The image is specified using its PATH directly.
        #
        # DO NOT pass images=[...] separately to
        # apply_chat_template().
        #
        # This lets SmolVLM/Idefics3 create:
        #
        # input_ids
        # attention_mask
        # pixel_values
        # pixel_attention_mask
        #
        # correctly.
        # ----------------------------------------------------

        messages = [

            {
                "role": "system",

                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },

            {
                "role": "user",

                "content": [

                    {
                        "type": "image",
                        "path": image_path,
                    },

                    {
                        "type": "text",
                        "text": ex["question"],
                    },

                ],
            },

            {
                "role": "assistant",

                "content": [

                    {
                        "type": "text",
                        "text": ex["answer"],
                    }

                ],
            },

        ]


        # ----------------------------------------------------
        # NATIVE SMOLVLM MULTIMODAL PROCESSING
        # ----------------------------------------------------

        batch = self.processor.apply_chat_template(

            messages,

            add_generation_prompt=False,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",
        )


        # ----------------------------------------------------
        # VERIFY PIXEL ATTENTION MASK
        # ----------------------------------------------------

        if "pixel_attention_mask" not in batch:

            raise RuntimeError(
                "SmolVLM processor did not return "
                "pixel_attention_mask.\n"
                f"Returned keys: {list(batch.keys())}"
            )


        pixel_mask = batch["pixel_attention_mask"]


        # Idefics3 expects spatial dimensions here.
        #
        # The previous crash was:
        #
        # Dimension out of range
        #
        # because this tensor was 1D.
        #

        if pixel_mask.ndim < 3:

            raise RuntimeError(
                "INVALID pixel_attention_mask shape: "
                f"{tuple(pixel_mask.shape)}\n"
                "Expected batch + spatial dimensions."
            )


        # ----------------------------------------------------
        # BUILD LABELS
        # ----------------------------------------------------

        labels = batch["input_ids"].clone()


        # Ignore padding
        if self.pad_token_id is not None:

            labels[
                labels == self.pad_token_id
            ] = -100


        # Ignore image placeholder tokens
        if self.image_token_id is not None:

            labels[
                labels == self.image_token_id
            ] = -100


        batch["labels"] = labels


        return batch


# ============================================================
# 3. BUILD MODEL + LoRA
# ============================================================

def build_model(processor):

    print("Loading SmolVLM...")


    model = AutoModelForImageTextToText.from_pretrained(

        MODEL_ID,

        dtype=torch.float16,

        # Stable attention implementation for this setup
        _attn_implementation="eager",
    )


    # --------------------------------------------------------
    # FIX INVALID MODEL CONFIG PAD TOKEN
    #
    # Your log showed:
    #
    # pad_token_id = 128002
    #
    # while vocabulary is only 0..31999 according to the
    # Transformers warning.
    #
    # Use the tokenizer's actual pad token instead.
    # --------------------------------------------------------

    if processor.tokenizer.pad_token_id is not None:

        model.config.pad_token_id = (
            processor.tokenizer.pad_token_id
        )


    # --------------------------------------------------------
    # MEMORY OPTIMIZATION
    # --------------------------------------------------------

    model.config.use_cache = False

    model.gradient_checkpointing_enable()


    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(

        r=8,

        lora_alpha=16,

        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "v_proj",
        ],

        task_type="CAUSAL_LM",
    )


    model = get_peft_model(
        model,
        lora_config
    )


    model.print_trainable_parameters()


    return model


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "SmolVLM Construction Site Safety "
        "LoRA Training"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    raw_dataset = load_raw_dataset(
        DATA_JSON
    )


    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    eval_size = min(
        0.1,
        5 / len(raw_dataset)
    )


    split = raw_dataset.train_test_split(

        test_size=eval_size,

        seed=42,
    )


    train_dataset = split["train"]

    eval_dataset = split["test"]


    print(
        f"Train examples: {len(train_dataset)} | "
        f"Eval examples: {len(eval_dataset)}"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model(
        processor
    )


    # --------------------------------------------------------
    # COLLATOR
    # --------------------------------------------------------

    collate_fn = SafetyCollator(
        processor
    )


    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,


        # GTX 1650 4GB
        per_device_train_batch_size=1,

        per_device_eval_batch_size=1,


        # Effective batch size = 8
        gradient_accumulation_steps=8,


        # Small dataset
        num_train_epochs=6,


        learning_rate=2e-4,

        warmup_steps=0,


        # Log every optimizer step
        logging_steps=1,


        # Transformers 5.16.1
        eval_strategy="epoch",

        save_strategy="epoch",


        save_total_limit=2,


        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,


        # GTX 1650
        fp16=True,


        optim="adamw_torch",


        report_to="none",


        # VERY IMPORTANT
        #
        # Our dataset contains:
        # image_path
        # question
        # answer
        #
        # The collator consumes them.
        #
        remove_unused_columns=False,


        # Windows
        dataloader_num_workers=0,


        # Memory
        gradient_checkpointing=True,
    )


    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        data_collator=collate_fn,
    )


    # --------------------------------------------------------
    # START TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()


    trainer.train()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING MODEL")
    print("=" * 70)


    trainer.save_model(
        OUTPUT_DIR
    )


    processor.save_pretrained(
        OUTPUT_DIR
    )


    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"LoRA adapter + processor saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()