import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_PATH = "test.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(IMAGE_PATH).convert("RGB")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    _attn_implementation="eager",
).to(device)

questions = [
    "Is there a worker in a potentially dangerous position?",
    "Is any worker working at height or on an elevated structure?",
    "Are the workers wearing appropriate PPE?",
    "Is there anything in the scene that could cause a slip, trip, or fall?",
    "Overall, is the scene SAFE, WARNING, or DANGER? Explain why.",
]

for q in questions:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": q},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
        )

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    print("\nQUESTION:")
    print(q)

    print("\nANSWER:")
    print(answer)

    print("-" * 60)