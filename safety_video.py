"""
safety_video.py
Video -> YOLO (every frame) + fine-tuned SmolVLM (every N seconds)
      -> annotated video with SAFE / WARNING / DANGER banner + JSON log.

Usage (run from the project root, with the .venv active):
    python safety_video.py my_video.mp4
    python safety_video.py my_video.mp4 --max-seconds 20      # quick test
    python safety_video.py my_video.mp4 --every 3             # VLM every 3 s
"""

import argparse
import json
import os
import re
import textwrap

import cv2
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from ultralytics import YOLO

BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_MODEL = "smolvlm-safety-lora"
YOLO_WEIGHTS = "yolo11n.pt"

VLM_IMAGE_SIDE = 384          # same size used in training -> fits 4GB VRAM
TMP_FRAME = "_tmp_vlm_frame.jpg"

# BGR colours
COLORS = {
    "SAFE": (0, 170, 0),
    "WARNING": (0, 140, 255),
    "DANGER": (0, 0, 220),
    "UNKNOWN": (120, 120, 120),
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def load_models():
    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading processor + SmolVLM + LoRA adapter...")
    processor = AutoProcessor.from_pretrained(LORA_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        _attn_implementation="eager",
    ).to(device)
    model = PeftModel.from_pretrained(model, LORA_MODEL)
    model.eval()
    print("Models ready on", device)
    return yolo, processor, model


# ---------------------------------------------------------------------------
# VLM call on one frame
# ---------------------------------------------------------------------------
def ask_vlm(processor, model, frame_bgr, yolo_summary):
    # Shrink the frame, save it to a temp file, and pass it by path
    # (exactly how your working yolo_vlm.py / video_vlm.py do it).
    h, w = frame_bgr.shape[:2]
    scale = VLM_IMAGE_SIDE / max(h, w)
    if scale < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    cv2.imwrite(TMP_FRAME, frame_bgr)

    question = (
        "Analyze this construction-site frame. "
        f"YOLO detected: {yolo_summary}. "
        "Check for Working at Height, PPE compliance, and Slips/Trips/Falls hazards. "
        "Start your answer with one word: SAFE, WARNING, or DANGER. "
        "Then explain the visual evidence in one or two sentences."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You are a construction site safety inspector. "
                "Analyze worker safety using visual evidence."
            )}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": TMP_FRAME},
                {"type": "text", "text": question},
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
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    start = inputs["input_ids"].shape[1]
    return processor.batch_decode(out[:, start:], skip_special_tokens=True)[0].strip()


def parse_verdict(answer):
    """Return the EARLIEST of SAFE / WARNING / DANGER found in the answer."""
    best, best_pos = "UNKNOWN", None
    for label in ("SAFE", "WARNING", "DANGER"):
        m = re.search(rf"\b{label}\b", answer, flags=re.IGNORECASE)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = label, m.start()
    return best


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        label = f"{d['object']} {d['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)


def draw_banner(frame, verdict, explanation, t_sec):
    h, w = frame.shape[:2]
    scale = max(0.5, w / 1280 * 0.7)
    bar_h = int(46 * scale / 0.7)

    cv2.rectangle(frame, (0, 0), (w, bar_h), COLORS[verdict], -1)
    cv2.putText(frame, f"{verdict}   |   t = {t_sec:.1f}s", (12, int(bar_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    text = explanation.replace("\n", " ").encode("ascii", "ignore").decode()
    chars = max(20, int((w - 24) / (20 * scale)))
    lines = textwrap.wrap(text, chars)[:2]
    if lines:
        box_h = int(30 * scale / 0.7) * len(lines) + 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_h), (w, bar_h + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, line in enumerate(lines):
            y = bar_h + int(26 * scale / 0.7) * (i + 1)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale * 0.75, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="output video path")
    ap.add_argument("--every", type=float, default=2.0,
                    help="run the VLM every N seconds of video (default 2)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only process the first N seconds (for quick tests)")
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.out or os.path.join("output", f"{base}_safety.mp4")
    log_path = os.path.join("output", f"{base}_safety_log.json")

    yolo, processor, model = load_models()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(args.max_seconds * fps) if args.max_seconds else total

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vlm_every = max(1, int(fps * args.every))

    verdict, explanation = "UNKNOWN", "Waiting for first VLM analysis..."
    log = []
    idx = 0

    print(f"\nProcessing {args.video}  ({w}x{h}, {fps:.1f} fps)")
    print(f"VLM runs every {args.every}s (every {vlm_every} frames)\n")

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = idx / fps

        # --- YOLO on every frame ---
        res = yolo(frame, verbose=False, conf=0.4)[0]
        detections = []
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({
                "object": res.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": [x1, y1, x2, y2],
            })

        # --- VLM every N seconds ---
        if idx % vlm_every == 0:
            counts = {}
            for d in detections:
                counts[d["object"]] = counts.get(d["object"], 0) + 1
            summary = ", ".join(f"{n} {k}" for k, n in counts.items()) or "nothing"

            explanation = ask_vlm(processor, model, frame, summary)
            verdict = parse_verdict(explanation)
            log.append({"time_sec": round(t_sec, 1), "verdict": verdict,
                        "yolo": summary, "vlm_answer": explanation})
            print(f"[{t_sec:6.1f}s] {verdict:8s} | {explanation[:100]}")

        # --- draw + save ---
        draw_boxes(frame, detections)
        draw_banner(frame, verdict, explanation, t_sec)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    if os.path.exists(TMP_FRAME):
        os.remove(TMP_FRAME)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Video: {out_path}")
    print(f"Log:   {log_path}")


if __name__ == "__main__":
    main()