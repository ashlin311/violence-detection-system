import torch
import torch.nn as nn
from torchvision import models, transforms

from utils import get_total_frames, get_tsn_indices, extract_tsn_frames, frame_to_tensor, encode_image_to_base64
from masking import apply_human_mask
from

class TSNModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [B, K, 3, H, W]
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        x = self.backbone(x)          # [B*K, num_classes]
        x = x.view(B, K, -1)          # [B, K, num_classes]
        out = x.mean(dim=1)           # [B, num_classes]
        return out


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TSNModel(num_classes=2, pretrained=False)
model.load_state_dict(torch.load("model/tsn_masked_best.pth", map_location=device))
model.to(device)
model.eval()

# Validation transform
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def run_inference(video_path):
    total_frames = get_total_frames(video_path)
    if total_frames <= 0:
        raise ValueError("Uploaded video has no readable frames")

    indices = get_tsn_indices(total_frames, num_segments=8, mode="val")
    frames = extract_tsn_frames(video_path, indices)
    if not frames:
        raise ValueError("No frames could be extracted from uploaded video")

    tensors = []
    masked_frames = []
    original_frames = []
    person_count = 0

    for frame in frames:
        masked_frame, count = apply_human_mask(frame)
        person_count = max(person_count, count)
        original_frames.append(frame)
        masked_frames.append(masked_frame)
        tensor = frame_to_tensor(masked_frame, val_transform)
        tensors.append(tensor)

    input_tensor = torch.stack(tensors).unsqueeze(0).to(device)  # [1, 8, 3, 224, 224]

    with torch.no_grad():
        # Get per-frame logits before averaging
        B, K, C, H, W = input_tensor.shape
        flat = input_tensor.view(B * K, C, H, W)
        per_frame_logits = model.backbone(flat)  # [8, 2]
        per_frame_logits = per_frame_logits.view(B, K, -1)  # [1, 8, 2]

        avg_logits = per_frame_logits.mean(dim=1)  # [1, 2]
        probs = torch.softmax(avg_logits, dim=1)

    # Class mapping: 0 = Non Violent, 1 = Violent (matches trained model)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    label = "Violent" if pred_class == 1 else "Non Violent"

    # Most active frame: highest Fight class score (class 1)
    fight_scores = per_frame_logits[0, :, 1]  # [8]
    most_active_idx = torch.argmax(fight_scores).item()

    original_frame = original_frames[most_active_idx]
    masked_frame = masked_frames[most_active_idx]

    return {
        "prediction": label,
        "confidence": confidence,
        "most_active_frame": most_active_idx,
        "original_frame": original_frame,
        "masked_frame": masked_frame,
        "person_count": person_count,
    }
