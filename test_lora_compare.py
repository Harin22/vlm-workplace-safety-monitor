import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_MODEL = "smolvlm-safety-lora"
IMAGE_PATH = "dataset/images/height_006.png"

QUESTION = "Is the worker working safely with respect to height?"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("BASE MODEL vs LoRA MODEL")
print("=" * 70)

# --------------------------------------------------
# Load processor
# --------------------------------------------------
processor = AutoProcessor.from_pretrained(LORA_MODEL)

# --------------------------------------------------
# Load base model
# --------------------------------------------------
print("\nLoading base model...")

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    _attn_implementation="eager"
).to(device)

base_model.eval()

# --------------------------------------------------
# Prepare conversation
# --------------------------------------------------
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a construction site safety inspector. "
                    "Analyze the image and answer the safety question."
                )
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "path": IMAGE_PATH
            },
            {
                "type": "text",
                "text": QUESTION
            }
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
    if torch.is_tensor(v)
}

# --------------------------------------------------
# Function to generate
# --------------------------------------------------
def generate(model):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    return processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True
    )[0]


# --------------------------------------------------
# BASE MODEL
# --------------------------------------------------
print("\n" + "=" * 70)
print("BASE MODEL OUTPUT")
print("=" * 70)

base_output = generate(base_model)

print(base_output)


# --------------------------------------------------
# LOAD LoRA
# --------------------------------------------------
print("\nLoading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)

lora_model.eval()

print("\n" + "=" * 70)
print("FINE-TUNED LoRA MODEL OUTPUT")
print("=" * 70)

lora_output = generate(lora_model)

print(lora_output)


# --------------------------------------------------
# DONE
# --------------------------------------------------
print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)