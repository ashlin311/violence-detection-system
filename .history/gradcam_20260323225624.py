# gradcam.py

import cv2
import torch
import numpy as np
from utils import encode_image_to_base64


def generate_gradcam(model, masked_frame, device, target_class=1):
    """
    Generate a Grad-CAM heatmap overlay for a single frame.

    Args:
        model      : TSNModel instance (already loaded, eval mode)
        masked_frame: numpy BGR frame (the most active masked frame)
        device     : torch device
        target_class: 1 = Fight (Violent), 0 = NonFight

    Returns:
        base64-encoded JPEG string of the heatmap overlaid on the frame
        or None if something goes wrong
    """

    # ── 1. Preprocessing ────────────────────────────────────────────────────
    # Import here to avoid circular imports (val_transform lives in model.py)
    from model import val_transform

    # frame_to_tensor returns a [3, 224, 224] tensor
    from utils import frame_to_tensor
    tensor = frame_to_tensor(masked_frame, val_transform)  # [3, 224, 224]
    input_tensor = tensor.unsqueeze(0).to(device)          # [1, 3, 224, 224]
    input_tensor.requires_grad_(False)

    # ── 2. Hook storage ─────────────────────────────────────────────────────
    activations = {}
    gradients   = {}

    def forward_hook(module, input, output):
        activations["layer4"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t. the output of this layer
        gradients["layer4"] = grad_output[0].detach()

    # ── 3. Register hooks on layer4[-1] ─────────────────────────────────────
    target_layer  = model.backbone.layer4[-1]
    fwd_handle    = target_layer.register_forward_hook(forward_hook)
    bwd_handle    = target_layer.register_full_backward_hook(backward_hook)

    try:
        # ── 4. Forward pass ─────────────────────────────────────────────────
        model.eval()
        # We run through the full backbone (includes fc) so gradients exist
        output = model.backbone(input_tensor)   # [1, 2]
        # output[0, 1] = Fight logit

        # ── 5. Backward pass on target class score ──────────────────────────
        model.zero_grad()
        # Create a one-hot-like scalar for the target class
        class_score = output[0, target_class]
        class_score.backward()

        # ── 6. Compute Grad-CAM weights ─────────────────────────────────────
        # activations["layer4"] : [1, 512, 7, 7]
        # gradients["layer4"]   : [1, 512, 7, 7]

        acts  = activations["layer4"].squeeze(0)   # [512, 7, 7]
        grads = gradients["layer4"].squeeze(0)     # [512, 7, 7]

        # Global average pool gradients over spatial dims → [512] weights
        weights = grads.mean(dim=(1, 2))            # [512]

        # Weighted sum of activation maps → [7, 7]
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ── 7. ReLU + normalise ─────────────────────────────────────────────
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()

        # Avoid divide-by-zero when all activations are zero
        cam_max = cam_np.max()
        if cam_max > 0:
            cam_np = cam_np / cam_max
        # cam_np is now in [0, 1] range, shape [7, 7]

        # ── 8. Resize heatmap to original frame size ────────────────────────
        h, w = masked_frame.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── 9. Apply jet colormap ───────────────────────────────────────────
        # cv2.applyColorMap expects uint8 [0, 255]
        cam_uint8   = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        # heatmap_bgr: [H, W, 3] BGR

        # ── 10. Overlay on masked frame ─────────────────────────────────────
        # masked_frame is already BGR numpy
        overlay = cv2.addWeighted(masked_frame, 0.5, heatmap_bgr, 0.5, 0)

        # ── 11. Encode to base64 ────────────────────────────────────────────
        return encode_image_to_base64(overlay)

    except Exception as e:
        print(f"[Grad-CAM] Error: {e}")
        return None

    finally:
        # Always remove hooks to avoid memory leaks
        fwd_handle.remove()
        bwd_handle.remove()