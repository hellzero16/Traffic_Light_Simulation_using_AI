from ultralytics import YOLO
import cv2
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

model = YOLO("yolov8m.pt").to(device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
light_states = ["red", "yellow", "green"]
current_light = 0
last_change = time.time()
cycle_time = 5

print("🚦 Smart Traffic Light running... Press 'q' to quit.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    results = model(frame, stream=True, conf=0.6, iou=0.45, classes=[0], device=device)

    human_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            if label == "person" and conf >= 0.6:
                human_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if human_count > 2:
        cycle_time = 20
    elif human_count == 2:
        cycle_time = 10
    elif human_count == 1:
        cycle_time = 5
    else:
        cycle_time = 5

    if time.time() - last_change >= cycle_time:
        current_light = (current_light + 1) % 3
        last_change = time.time()

    light_color = colors[light_states[current_light]]

    cv2.rectangle(frame, (20, 50), (120, 250), (50, 50, 50), -1)
    for i, color in enumerate(["red", "yellow", "green"]):
        center = (70, 90 + i * 60)
        radius = 25
        if i == current_light:
            cv2.circle(frame, center, radius, colors[color], -1)
        else:
            cv2.circle(frame, center, radius, (100, 100, 100), -1)

    cv2.putText(frame, f"Humans: {human_count}", (160, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Cycle: {cycle_time}s", (160, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Light: {light_states[current_light].upper()}", (160, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, light_color, 3)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (160, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("🚦 Smart Traffic Light with YOLOv8 (GPU Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Program ended.")
