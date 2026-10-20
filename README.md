# VLM Workplace Safety Monitor

A lightweight **Vision-Language Model + YOLO** system that watches construction-site video and labels each moment as **SAFE / WARNING / DANGER**, with a short text explanation.

It uses **SmolVLM-500M-Instruct** fine-tuned with **LoRA** to reason about three safety categories:

1. **Working at Height**
2. **Site Entry / PPE Compliance**
3. **Slips, Trips & Falls**

The whole project is built to run on a small GPU with a local model (**GTX 1650, 4 GB VRAM**).

---

## How it works

```
Construction video
      |
      v
YOLO (every frame)  ->  people / objects + boxes
      |
      v
Fine-tuned SmolVLM (every N seconds)  ->  safety reasoning on the frame
      |
      v
SAFE / WARNING / DANGER banner + explanation drawn on the video
      |
      v
Annotated output video + JSON log
```

- **YOLO** is fast, so it runs on every frame and draws the boxes.
- **SmolVLM** is slower, so it runs every 2 seconds (configurable). Its latest verdict stays on screen until the next check.
- The VLM receives the frame plus a short summary of what YOLO found (e.g. "2 person").

---

## Demo

<!-- Add a screenshot or GIF of output/<your_video>_safety.mp4 here, e.g.
![Demo](assets/demo.png) -->

---

## Quick start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA build
pip install transformers peft datasets pillow ultralytics
```

(`ultralytics` also installs OpenCV.) YOLO weights (`yolo11n.pt`) download automatically on first run.

### 2. Run on a video

Put a video in a `videos/` folder, then:

```bash
python safety_video.py videos/my_video.mp4
python safety_video.py videos/my_video.mp4 --max-seconds 20   # quick test
python safety_video.py videos/my_video.mp4 --every 3          # VLM every 3 s
```

Outputs are saved in `output/`:
- `<name>_safety.mp4` - video with YOLO boxes and the SAFE / WARNING / DANGER banner
- `<name>_safety_log.json` - timestamp, verdict, YOLO summary and the VLM's answer for each check

### 3. Retrain (optional)

```bash
python train_lora.py
```

Reads `dataset/train.json` and `dataset/images/`, and saves the LoRA adapter to `smolvlm-safety-lora/`.

---

## Training details

| Item | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Method | LoRA (r=8, alpha=16, dropout=0.05) |
| Target modules | `q_proj`, `v_proj` |
| Trainable parameters | ~1.1M of ~508M (0.22%) |
| Training images | 46 |
| Precision | fp16 |
| Memory tricks | batch size 1 + gradient accumulation, gradient checkpointing, images resized to 384 px |
| Hardware | NVIDIA GTX 1650 (4 GB) |

**Key engineering note:** SmolVLM's image processor produces ragged, nested tensors that HuggingFace `datasets` stores as plain lists, which crashes custom `pad_sequence` collators. The fix was to keep the dataset as raw strings (image path, question, answer) and do all tokenization and image processing inside the collate function, one example at a time, using the processor's own chat template.

---

## Project structure

```
dataset/                 training images + train.json
smolvlm-safety-lora/     fine-tuned LoRA adapter

train_lora.py            LoRA fine-tuning script
check_dataset.py         dataset sanity checks

extract_frames.py        video -> frames
yolo_vid.py              YOLO-only video detection
yolo_vlm.py              YOLO + VLM on extracted frames (text output)
video_vlm.py             VLM on extracted frames (text output)
safety_video.py          MAIN DEMO: video -> annotated video with verdicts

test_finetuned.py        test the fine-tuned model
test_lora_compare.py     base model vs LoRA comparison
test_6_images.py         quick multi-image test
safety_baseline.py       base-model baseline
test_vlm.py, test_vqa.py early experiments
experiments.md           experiment notes
app.py                   app entry point
```

---

## Limitations

- **Small dataset (46 images).** The model's verdicts are conservative and often short (e.g. just "SAFE."). It needs a larger, balanced dataset with clear SAFE / WARNING / DANGER examples.
- **No quantitative evaluation yet.** I have not measured accuracy, so I don't report any.
- **YOLO is the generic COCO model (`yolo11n`).** It detects people and common objects, but not helmets, vests, harnesses or ladders specifically.
- **The VLM looks at one frame at a time**, every N seconds. There is no real temporal reasoning across frames yet.
- **Speed:** the VLM is not real-time on a GTX 1650, so processing is offline.

## Next steps

- Grow and balance the dataset (more WARNING / DANGER cases per category)
- Train or fine-tune a YOLO model for PPE (helmet, vest, harness)
- Add temporal analysis across frames (e.g. consistent hazard for several seconds before raising DANGER)
- Evaluate on a held-out set and report accuracy per category

---

## License

MIT
