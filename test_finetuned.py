import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_MODEL = "smolvlm-safety-lora"

IMAGE_PATH = "dataset/images/height_006.png"


print("=" * 70)
print("Fine-tuned SmolVLM - Image Test")
print("=" * 70)

# --------------------------------------------------
# 1. Load processor
# --------------------------------------------------
print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(LORA_MODEL)


# --------------------------------------------------
# 2. Load base model
# --------------------------------------------------
print("Loading base SmolVLM...")

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    _attn_implementation="eager"
)


# --------------------------------------------------
# 3. Load our LoRA adapter
# --------------------------------------------------
print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)

model.eval()


# --------------------------------------------------
# 4. Load image
# --------------------------------------------------
print(f"Loading image: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")


# --------------------------------------------------
# 5. Create conversation
# --------------------------------------------------
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a construction site safety inspector. "
                    "Analyze the image and classify the situation as "
                    "SAFE, WARNING, or DANGER. Focus on Working at Height, "
                    "Site Entry/PPE Compliance, and Slips/Trips/Falls. "
                    "Give a short explanation based on visible evidence."
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
                "text": (
                    "Analyze this construction site image. "
                    "What is the safety status and why?"
                )
            }
        ],
    },
]


# --------------------------------------------------
# 6. Prepare model input
# --------------------------------------------------
print("Preparing input...")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# Move tensors to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
    if torch.is_tensor(value)
}

model = model.to(device)


# --------------------------------------------------
# 7. Generate response
# --------------------------------------------------
print("\nRunning inference...")
print("-" * 70)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )


# --------------------------------------------------
# 8. Decode response
# --------------------------------------------------
input_length = inputs["input_ids"].shape[1]

generated_text = processor.batch_decode(
    generated_ids[:, input_length:],
    skip_special_tokens=True
)[0]


print("\nMODEL RESPONSE:")
print(generated_text)

print("\n" + "=" * 70)
print("IMAGE TEST COMPLETE")
print("=" * 70)