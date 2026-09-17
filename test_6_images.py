import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_MODEL = "smolvlm-safety-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

tests = [
    ("height_005.png", "Is the worker working safely with respect to height?"),
    ("height_006.png", "Is the worker working safely with respect to height?"),

    ("ppe_001.png", "Is the worker following the required PPE safety rules?"),
    ("ppe_002.png", "Is the worker following the required PPE safety rules?"),

    ("fall_001.png", "Is there a slip, trip, or fall hazard in this scene?"),
    ("fall_002.png", "Is there a slip, trip, or fall hazard in this scene?"),
]

processor = AutoProcessor.from_pretrained(LORA_MODEL)

model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    _attn_implementation="eager"
).to(device)

model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()

for image_name, question in tests:

    print("\n" + "=" * 70)
    print(f"IMAGE: {image_name}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a construction site safety inspector. "
                        "Identify the relevant safety hazard and classify "
                        "the situation as SAFE, WARNING, or DANGER. "
                        "Focus on the specific hazard mentioned in the question."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": f"dataset/images/{image_name}"
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
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

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[1]

    answer = processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True
    )[0]

    print("\nMODEL:")
    print(answer)