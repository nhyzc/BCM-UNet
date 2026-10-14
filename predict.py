"""Single-image crack segmentation with the variant selected in config.py.

The preprocessing follows the batch pipeline: an optional circular mask removes
the region outside the coal core before inference, and a median filter cleans
the predicted boundary afterwards.

Usage:
    python predict.py --image path/to/slice.png
    python predict.py --image slice.png --model output_results/BoundaryUNet_SR111/best_model.pth
"""
import argparse
import math
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from config import Config
from models import build_model, load_checkpoint


def detect_circular_region(img_np, threshold_val=50, safe_margin=10):
    """Locate the circular coal-core region inside a square CT slice."""
    h, w = img_np.shape

    border_mask = np.zeros_like(img_np)
    border_width = 50
    border_mask[border_width:h - border_width, border_width:w - border_width] = 255
    img_masked = cv2.bitwise_and(img_np, img_np, mask=border_mask)

    _, binary = cv2.threshold(img_masked, threshold_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('Warning: no circular region detected, using the full image')
        return None, None, None

    max_cnt = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(max_cnt)

    img_center_x, img_center_y = w // 2, h // 2
    if math.sqrt((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2) > 100:
        cx, cy = img_center_x, img_center_y

    return int(cx), int(cy), int(radius - safe_margin)


def create_circular_mask(h, w, cx, cy, radius):
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2).astype(np.uint8)


def predict_single_image(model, image_path, output_dir, threshold=0.5,
                         use_circular_mask=True, threshold_val=50, safe_margin=30):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    img_pil = Image.open(image_path).convert('L')
    original_size = img_pil.size
    img_np = np.array(img_pil)

    circular_mask = None
    cx = cy = safe_radius = None

    if use_circular_mask:
        cx, cy, safe_radius = detect_circular_region(img_np, threshold_val, safe_margin)
        if cx is not None:
            print('Detected circular region: center=({}, {}), radius={}'.format(cx, cy, safe_radius))
            circular_mask = create_circular_mask(img_np.shape[0], img_np.shape[1], cx, cy, safe_radius)
            bg_color = 79
            img_np = img_np * circular_mask + bg_color * (1 - circular_mask)
            img_pil = Image.fromarray(img_np)

    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
    ])

    # The model is placed on the inference device.
    model = model.to(Config.DEVICE)
    img_tensor = transform(img_pil).unsqueeze(0).to(Config.DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = (pred_prob > threshold).astype(np.uint8)

    if circular_mask is not None:
        mask_resized = cv2.resize(circular_mask, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))
        binary_mask = binary_mask * mask_resized

    mask_img = Image.fromarray(binary_mask * 255).convert('L')
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_pil, cmap='gray')
    if cx is not None and safe_radius is not None:
        plt.gca().add_patch(plt.Circle((cx, cy), safe_radius, fill=False, color='red', linewidth=2))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_prob, cmap='jet')
    plt.colorbar(label='Probability')
    plt.title('Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    img_vis = np.array(img_pil.resize(Config.IMG_SIZE)) / 255.0
    overlay_mask = np.array(mask_img.resize(Config.IMG_SIZE)) / 255.0
    colored = np.zeros((*Config.IMG_SIZE, 3))
    colored[..., 0] = overlay_mask
    overlay = np.expand_dims(img_vis, axis=2) * 0.7 + colored * 0.3
    plt.imshow(np.clip(overlay, 0, 1))
    plt.title('Overlay')
    plt.axis('off')

    save_path = os.path.join(output_dir, '{}_result.png'.format(base_name))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print('Result saved to: {}'.format(save_path))
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=str, required=True, help='input image path')
    parser.add_argument('--model', type=str, default=None,
                        help='checkpoint path (default: Config.BEST_MODEL_PATH)')
    parser.add_argument('--output', type=str, default=None,
                        help='output directory (default: <run dir>/predictions)')
    parser.add_argument('--threshold', type=float, default=0.5, help='binarisation threshold')
    parser.add_argument('--no-circular-mask', action='store_true',
                        help='disable circular coal-core detection')
    parser.add_argument('--threshold_val', type=int, default=50,
                        help='threshold used for circular region detection')
    parser.add_argument('--safe_margin', type=int, default=30,
                        help='safety margin (px) applied to the detected radius')
    args = parser.parse_args()

    model_path = args.model or Config.BEST_MODEL_PATH
    output_dir = args.output or os.path.join(Config.OUTPUT_DIR, 'predictions')
    if not os.path.exists(model_path):
        raise SystemExit(
            'Checkpoint not found: {}\n'
            'Train a model first, or pass --model explicitly.'.format(model_path)
        )

    print('Loading {} from {} on {}'.format(Config.MODEL_NAME, model_path, Config.DEVICE))
    model = build_model(Config)
    load_checkpoint(model, model_path, Config.DEVICE)

    predict_single_image(
        model, args.image, output_dir,
        threshold=args.threshold,
        use_circular_mask=not args.no_circular_mask,
        threshold_val=args.threshold_val,
        safe_margin=args.safe_margin,
    )


if __name__ == '__main__':
    main()
