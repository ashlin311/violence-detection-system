import cv2
import base64
import numpy as np
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(total, 0)


def get_tsn_indices(total_frames, num_segments, mode="val"):
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")

    segment_len = total_frames / num_segments if total_frames > 0 else 0
    indices = []

    for i in range(num_segments):
        start = int(i * segment_len)
        end = int((i + 1) * segment_len)

        if mode == "train":
            if end > start:
                idx = int(np.random.randint(start, end))
            else:
                idx = start
        else:
            # Validation/test uses deterministic middle-frame sampling per segment.
            idx = (start + end) // 2

        if total_frames > 0:
            idx = min(max(idx, 0), total_frames - 1)
        else:
            idx = 0

        indices.append(idx)

    return indices


def extract_tsn_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    frames = []

    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx}")
            frames.append(frame)
    finally:
        cap.release()

    return frames


def frame_to_tensor(frame, transform=None):
    if len(frame.shape) == 2:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transform(rgb_frame)


def encode_image_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')
