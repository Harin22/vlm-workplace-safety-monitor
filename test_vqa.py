import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_PATH = "test.png"


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


image = Image.open(IMAGE_PATH).convert("RGB")
print("Image loaded.")


print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Processor loaded.")


print("\nLoading SmolVLM...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    _attn_implementation="eager",
)

model = model.to(device)

print("Model loaded successfully.")


print("\nSmolVLM Interactive VQA")
print("Ask questions about the image.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower().strip() == "exit":
        print("Exiting...")
        break

    if not question.strip():
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
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

    print("\nVLM is thinking...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
        )

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    print("\nSmolVLM:", answer)
    print()