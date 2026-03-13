import cv2
import base64
import numpy as np
import torch
from torchvision import transforms


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def get_tsn_indices(total_frames, num_segments, mode="val"):
    segment_len = total_frames / num_segments
    indices = []
    for i in range(num_segments):
        start = int(i * segment_len)
        end = int((i + 1) * segment_len)
        if mode == "val":
            mid = (start + end) // 2
            indices.append(mid)
        else:
            indices.append(start)
    return indices


def extract_tsn_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def frame_to_tensor(frame, transform):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb_frame)
    return tensor


def encode_image_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')
