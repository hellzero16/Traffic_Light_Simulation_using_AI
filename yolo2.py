from ultralytics import YOLO
import cv2
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "yolov8m.pt"
model = YOLO(model_path).to(device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

colors = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0)
}

light_states = ["red", "yellow", "green"]
current_light = 0
last_light_switch = time.time()
default_cycle_duration = 5
current_cycle_duration = default_cycle_duration
font = cv2.FONT_HERSHEY_SIMPLEX

print("Smart Traffic Light Initialized. Press 'q' to quit.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    detections = model(frame, stream=True, conf=0.6, iou=0.45, classes=[0], device=device)

    human_count = 0
    for result in detections:
        for box in result.boxes:
            class_index = int(box.cls[0])
            class_label = model.names[class_index]
            confidence = float(box.conf[0])
            if class_label == "person" and confidence >= 0.6:
                human_count += 1
                coords = list(map(int, box.xyxy[0]))
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_label} {confidence:.2f}", (x1, y1 - 10),
                            font, 0.6, (0, 255, 0), 2)

    if human_count > 2:
        current_cycle_duration = 20
    elif human_count == 2:
        current_cycle_duration = 10
    elif human_count == 1:
        current_cycle_duration = 5
    else:
        current_cycle_duration = default_cycle_duration

    current_time = time.time()
    time_elapsed = current_time - last_light_switch
    if time_elapsed >= current_cycle_duration:
        current_light = (current_light + 1) % len(light_states)
        last_light_switch = current_time

    current_color_name = light_states[current_light]
    current_color = colors[current_color_name]

    traffic_box_top_left = (20, 50)
    traffic_box_bottom_right = (120, 250)
    cv2.rectangle(frame, traffic_box_top_left, traffic_box_bottom_right, (50, 50, 50), -1)

    for i, color_name in enumerate(["red", "yellow", "green"]):
        center = (70, 90 + i * 60)
        radius = 25
        if i == current_light:
            cv2.circle(frame, center, radius, colors[color_name], -1)
        else:
            cv2.circle(frame, center, radius, (100, 100, 100), -1)

    text_color = (255, 255, 255)
    cv2.putText(frame, f"Detected Humans: {human_count}", (160, 80), font, 0.8, text_color, 2)
    cv2.putText(frame, f"Cycle Duration: {current_cycle_duration}s", (160, 120), font, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Current Light: {current_color_name.upper()}", (160, 160), font, 0.9, current_color, 3)

    fps = 1.0 / (time.time() - start_time)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (160, 200), font, 0.7, text_color, 2)

    height, width, _ = frame.shape
    cv2.putText(frame, "Smart AI-Based Traffic Controller", (width - 430, height - 20), font, 0.6, (200, 255, 200), 2)
    cv2.putText(frame, f"Device: {device.upper()}", (width - 220, 40), font, 0.6, (180, 180, 255), 2)

    current_timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, f"Time: {current_timestamp}", (width - 220, 70), font, 0.6, (255, 200, 180), 2)

    cv2.imshow("Smart Traffic Light with YOLOv8", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('r'):
        print("Resetting to default cycle duration")
        current_cycle_duration = default_cycle_duration
        current_light = 0
        last_light_switch = time.time()

cap.release()
cv2.destroyAllWindows()
print("Program Terminated Successfully.")
