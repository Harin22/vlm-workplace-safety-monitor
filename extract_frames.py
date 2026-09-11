import cv2
import os

VIDEO_PATH = "videos/test_vid.mp4"
OUTPUT_DIR = "frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % int(fps) == 0:
        filename = f"{OUTPUT_DIR}/frame_{frame_number:04d}.jpg"
        cv2.imwrite(filename, frame)

    frame_number += 1

cap.release()

print(f"Extracted frames to: {OUTPUT_DIR}")