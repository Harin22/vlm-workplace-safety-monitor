import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from ultralytics import YOLO

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_MODEL = "smolvlm-safety-lora"
FRAMES_DIR = "frames"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("YOLO + FINE-TUNED SmolVLM — VIDEO SAFETY REASONING")
print("=" * 70)

print("\nDevice:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# LOAD YOLO
# ---------------------------------------------------------

print("\nLoading YOLO...")

yolo = YOLO("yolo11n.pt")

print("YOLO loaded.")


# ---------------------------------------------------------
# LOAD VLM
# ---------------------------------------------------------

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(LORA_MODEL)

print("Loading SmolVLM...")

model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    _attn_implementation="eager",
).to(device)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    LORA_MODEL
)

model.eval()

print("Fine-tuned VLM loaded.")


# ---------------------------------------------------------
# GET FRAMES
# ---------------------------------------------------------

frame_files = sorted(
    file for file in os.listdir(FRAMES_DIR)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
)

print(f"\nFound {len(frame_files)} frames.")


# ---------------------------------------------------------
# PROCESS EACH FRAME
# ---------------------------------------------------------

for frame_file in frame_files:

    image_path = os.path.join(FRAMES_DIR, frame_file)

    print("\n" + "=" * 70)
    print("FRAME:", frame_file)
    print("=" * 70)

    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

    results = yolo(image_path, verbose=False)

    result = results[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "object": class_name,
            "confidence": round(confidence, 2),
            "box": [
                round(x1),
                round(y1),
                round(x2),
                round(y2)
            ]
        })

    print("\nYOLO:")
    
    if detections:
        for detection in detections:
            print(detection)
    else:
        print("No objects detected.")


    # -----------------------------------------------------
    # CREATE YOLO CONTEXT FOR VLM
    # -----------------------------------------------------

    if detections:

        detection_text = "\n".join(
            f"- {d['object']} "
            f"(confidence {d['confidence']}, "
            f"location {d['box']})"
            for d in detections
        )

    else:

        detection_text = "No YOLO objects detected."


    # -----------------------------------------------------
    # VLM PROMPT
    # -----------------------------------------------------

    question = f"""
Analyze this construction-site frame using the actual image
and the YOLO perception information below.

YOLO detections:
{detection_text}

You are the safety reasoning system.

Determine:

1. What is the worker doing?
2. What is happening in the scene?
3. Is there a potential safety hazard?
4. Which of these categories is relevant?
   - Working at Height
   - PPE Compliance
   - Slips, Trips & Falls
5. Classify the situation as:
   SAFE, WARNING, or DANGER
6. Explain the visual evidence supporting your judgment.

Do not simply list detected objects.
Reason about the worker's activity and the surrounding context.
"""


    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a construction-site safety inspector. "
                        "Use visual reasoning to understand worker "
                        "activities, hazards, and safety conditions."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": image_path
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]


    # -----------------------------------------------------
    # PROCESS
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
    # VLM GENERATION
    # -----------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )


    input_length = inputs["input_ids"].shape[1]

    answer = processor.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
    )[0]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    print("\nVLM SAFETY REASONING:")
    print(answer)


print("\n" + "=" * 70)
print("VIDEO ANALYSIS COMPLETE")
print("=" * 70)