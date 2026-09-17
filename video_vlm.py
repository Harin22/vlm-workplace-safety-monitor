import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_MODEL = "smolvlm-safety-lora"
FRAMES_DIR = "frames"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("VLM CONSTRUCTION SITE SAFETY - VIDEO ANALYSIS")
print("=" * 70)

print("\nDevice:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------
# LOAD PROCESSOR
# ---------------------------------------------------------

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(LORA_MODEL)

# ---------------------------------------------------------
# LOAD BASE MODEL
# ---------------------------------------------------------

print("Loading base SmolVLM...")

model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    _attn_implementation="eager",
).to(device)

# ---------------------------------------------------------
# LOAD LoRA ADAPTER
# ---------------------------------------------------------

print("Loading fine-tuned LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL
)

model.eval()

print("Fine-tuned model loaded successfully.")

# ---------------------------------------------------------
# SAFETY PROMPT
# ---------------------------------------------------------

question = (
    "You are a construction site safety inspector. "
    "Analyze this scene carefully. Focus on worker safety. "
    "Identify whether there is a potential hazard involving "
    "working at height, PPE compliance, or slips, trips and falls. "
    "Respond with the safety status and briefly explain the visual evidence."
)

# ---------------------------------------------------------
# FIND FRAMES
# ---------------------------------------------------------

frame_files = sorted(
    file
    for file in os.listdir(FRAMES_DIR)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
)

print(f"\nFound {len(frame_files)} frames.")

# ---------------------------------------------------------
# ANALYZE EACH FRAME
# ---------------------------------------------------------

for frame_file in frame_files:

    image_path = os.path.join(FRAMES_DIR, frame_file)

    print("\n" + "=" * 70)
    print("FRAME:", frame_file)
    print("=" * 70)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a construction site safety inspector. "
                        "Analyze worker safety using visual evidence."
                    ),
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
                    "text": question,
                },
            ],
        },
    ]

    # -----------------------------------------------------
    # PROCESS IMAGE + PROMPT
    # -----------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if torch.is_tensor(value)
    }

    # -----------------------------------------------------
    # GENERATE RESPONSE
    # -----------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Only decode newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    answer = processor.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
    )[0]

    print("\nVLM:")
    print(answer)

print("\n" + "=" * 70)
print("VIDEO ANALYSIS COMPLETE")
print("=" * 70)