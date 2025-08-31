import cv2
import numpy as np
import os
import threading
import time
from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolov8x.pt")

frame_lock = threading.Lock()
current_frame = None
detections = []


def camera_thread():
    global current_frame
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            current_frame = frame.copy()
        time.sleep(0.01)  # Prevent CPU overuse
    cap.release()


def detection_thread():
    global current_frame, detections

    while True:
        if current_frame is None:
            time.sleep(0.01)
            continue

        with frame_lock:
            frame = current_frame.copy()

        results = model.predict(frame, verbose=False)

        local_detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf > 0.5:
                    label_text = model.names[int(box.cls)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    local_detections.append((x1, y1, x2, y2, conf, label_text))

        detections = local_detections


def main():
    global detections, current_frame

    t1 = threading.Thread(target=camera_thread, daemon=True)
    t2 = threading.Thread(target=detection_thread, daemon=True)
    t1.start()
    t2.start()

    while True:
        if current_frame is None:
            continue

        frame = current_frame.copy()

        # Draw detections
        for x1, y1, x2, y2, conf, label_text in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
