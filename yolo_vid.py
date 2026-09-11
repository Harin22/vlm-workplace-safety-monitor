import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "videos/test_vid.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

print("Video opened successfully.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()