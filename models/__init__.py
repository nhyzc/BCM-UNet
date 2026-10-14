"""Model factory shared by the training and prediction scripts.

Keeping model construction here means that changing the switches in config.py
is enough to train, predict with, or evaluate any variant.
"""
import torch

from .unet import UNet
from .boundary_unet import BoundaryUNet

__all__ = ['UNet', 'BoundaryUNet', 'build_model', 'load_checkpoint']


def build_model(cfg):
    """Instantiate the model selected by the given Config object."""
    if cfg.MODEL_NAME == 'UNet':
        return UNet(in_channels=1, out_channels=1)
    if cfg.MODEL_NAME == 'BoundaryUNet':
        return BoundaryUNet(
            in_channels=1,
            out_channels=1,
            use_boundary=cfg.USE_BOUNDARY_MODULE,
            use_continuity=cfg.USE_CONTINUITY_MODULE,
            use_multiscale=cfg.USE_MULTISCALE_FUSION,
        )
    raise ValueError(
        "Unknown MODEL_NAME: {!r} (expected 'UNet' or 'BoundaryUNet')".format(cfg.MODEL_NAME)
    )


def load_checkpoint(model, path, device):
    """Load a state dict into the model and return it in eval mode."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
