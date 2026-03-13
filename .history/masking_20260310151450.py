import numpy as np
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")


def apply_human_mask(frame, yolo_model, dim_factor=0.3):
    results = yolo_model(frame, verbose=False)
    mask = np.full(frame.shape[:2], dim_factor, dtype=np.float32)
    person_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask[y1:y2, x1:x2] = 1.0

    masked_frame = (frame * mask[:, :, np.newaxis]).astype(np.uint8)
    return masked_frame, person_count
