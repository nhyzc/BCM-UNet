"""BCM-UNet inference for the desktop application."""
import numpy as np
from PIL import Image
import os, sys

# PyInstaller on Windows may locate c10.dll but fail to resolve its dependent
# CUDA/OpenMP runtime DLLs. Register the bundled directories before importing
# any model code. This keeps the executable independent of the user's PATH.
if getattr(sys, 'frozen', False):
    _BUNDLE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    _DLL_DIRS = [
        os.path.join(_BUNDLE_DIR, 'torch', 'lib'),
        _BUNDLE_DIR,
    ]
    for _dll_dir in _DLL_DIRS:
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except (AttributeError, OSError):
                pass

import torch

# In frozen mode the model is in _internal; in source mode it is in the project
# root, at the location defined by config.py.
if getattr(sys, 'frozen', False):
    _BASE = sys._MEIPASS  # PyInstaller resource directory.
else:
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _BASE not in sys.path:
        sys.path.insert(0, _BASE)

_MODEL_PATH = os.path.join(_BASE, 'best_model.pth')
if not os.path.exists(_MODEL_PATH):
    try:
        from config import Config as _Config
        _MODEL_PATH = _Config.BEST_MODEL_PATH
    except Exception:
        _MODEL_PATH = os.path.join(_BASE, 'output_results',
                                   'BoundaryUNet_SR111', 'best_model.pth')
IMG_SIZE = (512, 512)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = None


def pil_to_tensor(img_pil):
    """Convert a PIL image to a (C,H,W) float tensor in [0, 1]."""
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_model():
    global _model
    if _model is not None:
        return _model

    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base not in sys.path:
        sys.path.insert(0, base)

    from models.boundary_unet import BoundaryUNet
    model = BoundaryUNet(in_channels=1, out_channels=1, bilinear=False,
                         use_boundary=True, use_continuity=True,
                         use_multiscale=True)
    state = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE); model.eval()
    _model = model
    return _model


def predict(image_path, threshold=0.5):
    """
    Single-image inference:
    PIL -> resize(512) -> tensor -> model -> sigmoid -> binary -> resize back
    """
    model = load_model()
    img_pil = Image.open(image_path).convert('L')
    original_size = img_pil.size

    # Resize → tensor → batch
    img_resized = img_pil.resize(IMG_SIZE, Image.BILINEAR)
    img_tensor = pil_to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        logits = output[0] if isinstance(output, tuple) else output
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Binary mask → resize to original
    binary = (pred_prob > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary).convert('L')
    mask_img = mask_img.resize(original_size, Image.NEAREST)

    # Probability map → resize to original
    prob_uint8 = (pred_prob * 255).astype(np.uint8).clip(0, 255)
    prob_img = Image.fromarray(prob_uint8).convert('L')
    prob_img = prob_img.resize(original_size, Image.NEAREST)

    return np.array(mask_img) > 127, np.array(prob_img) / 255.0
