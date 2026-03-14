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
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Video has no decodable frames")

        if not frame_indices:
            return frames

        clamped_indices = [
            min(max(int(idx), 0), total_frames - 1)
            for idx in frame_indices
        ]
        index_to_positions = {}
        for pos, idx in enumerate(clamped_indices):
            index_to_positions.setdefault(idx, []).append(pos)

        sorted_targets = sorted(index_to_positions.keys())
        resolved = [None] * len(clamped_indices)

        target_ptr = 0
        current_idx = -1
        first_valid_frame = None
        last_valid_frame = None
        consecutive_failures = 0

        # Sequential decode is more robust than random seeking for WebM/VP8/VP9.
        while target_ptr < len(sorted_targets):
            ret, candidate = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    break
                continue

            consecutive_failures = 0
            current_idx += 1
            if candidate is not None and candidate.size > 0:
                if first_valid_frame is None:
                    first_valid_frame = candidate.copy()
                last_valid_frame = candidate

            while target_ptr < len(sorted_targets) and sorted_targets[target_ptr] <= current_idx:
                target_idx = sorted_targets[target_ptr]
                fill_frame = last_valid_frame if last_valid_frame is not None else first_valid_frame
                if fill_frame is not None:
                    for out_pos in index_to_positions[target_idx]:
                        resolved[out_pos] = fill_frame.copy()
                target_ptr += 1

        if first_valid_frame is None:
            raise RuntimeError("Uploaded video could not be decoded")

        fallback_frame = last_valid_frame if last_valid_frame is not None else first_valid_frame
        for i, frame in enumerate(resolved):
            if frame is None:
                resolved[i] = fallback_frame.copy()

        frames = resolved
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
