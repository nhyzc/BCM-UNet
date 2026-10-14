"""Central configuration and ablation control panel for BCM-UNet.

This is the ONLY file that needs editing in order to train, predict with, or
evaluate a model variant. Set MODEL_NAME and the three module switches, then
run ``python train.py``. Each variant writes to its own output directory, so
runs never overwrite one another.

Every path is derived from the location of this file, so the project runs from
any working directory.
"""
import os
import torch

# Directory containing this file; the anchor for every derived path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class _ConfigMeta(type):
    """Reject unknown option names.

    The switches are assigned on the class object itself, as in
    ``Config.USE_BOUNDARY_MODULE = True``, so the check lives on the metaclass:
    a plain ``__setattr__`` defined on Config would never be called.
    """

    def __setattr__(cls, name, value):
        if not hasattr(cls, name):
            raise AttributeError(
                "Unknown configuration option '{}'. Check the spelling against "
                'config.py.'.format(name))
        super().__setattr__(name, value)


class Config(metaclass=_ConfigMeta):
    # ==================================================================
    # Paths (resolved relative to this file, never hard-coded)
    # ==================================================================
    # Training dataset root. Defaults to a sibling directory named
    # "augmented" containing images/ and labels/, i.e. <parent>/augmented.
    DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, os.pardir, 'augmented'))
    IMAGE_DIR = os.path.join(DATA_ROOT, 'images')
    MASK_DIR = os.path.join(DATA_ROOT, 'labels')

    # Root directory for checkpoints, logs and prediction figures.
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'output_results')

    # ==================================================================
    # Ablation control panel
    # ==================================================================
    # 1. Backbone: 'UNet' (plain baseline) or 'BoundaryUNet' (proposed model).
    MODEL_NAME = 'BoundaryUNet'

    # 2. Boundary module: edge prediction + boundary attention, supervised by
    #    the joint edge + segmentation loss. The architecture and its loss are
    #    two halves of one module, so this single switch controls both.
    USE_BOUNDARY_MODULE = True

    # 3. Continuity enhancement module (dilated multi-receptive-field block).
    USE_CONTINUITY_MODULE = True

    # 4. Multi-scale fusion decoder.
    USE_MULTISCALE_FUSION = True

    # Run directory name under OUTPUT_ROOT. Leave as None to derive it from the
    # switches above (recommended), or set an explicit string to name a run.
    EXP_NAME = None

    # ==================================================================
    # Dataset split
    # ==================================================================
    # The dataset is shuffled with a fixed seed and divided 7:1:2 into training,
    # validation and test sets. The training set fits the model, the validation
    # set selects the best checkpoint after each epoch, and the test set is held
    # back for the final metric comparison.
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.2
    SEED = 42

    # ==================================================================
    # Data loading and memory
    # ==================================================================
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 1
    NUM_WORKERS = 4

    # ==================================================================
    # Optimisation
    # ==================================================================
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================================================================
    # Derived paths. refresh() recomputes them from the switches above.
    # ==================================================================
    exp_name = None
    OUTPUT_DIR = None
    BEST_MODEL_PATH = None
    LOG_FILE = None

    @classmethod
    def refresh(cls):
        """Recompute every path derived from the switches above.

        Call this after modifying MODEL_NAME, the module switches or EXP_NAME,
        and before train() or any prediction script reads the paths.
        """
        if cls.EXP_NAME:
            cls.exp_name = cls.EXP_NAME
        elif cls.MODEL_NAME == 'UNet':
            cls.exp_name = 'UNet_Baseline'
        else:
            cls.exp_name = 'BoundaryUNet_SR{}{}{}'.format(
                int(cls.USE_BOUNDARY_MODULE),
                int(cls.USE_CONTINUITY_MODULE),
                int(cls.USE_MULTISCALE_FUSION),
            )

        cls.OUTPUT_DIR = os.path.join(cls.OUTPUT_ROOT, cls.exp_name)
        cls.BEST_MODEL_PATH = os.path.join(cls.OUTPUT_DIR, 'best_model.pth')
        cls.LOG_FILE = os.path.join(cls.OUTPUT_DIR, 'training_log.csv')
        return cls.exp_name

    @classmethod
    def setup_dirs(cls):
        """Create the output directory for the current run."""
        cls.refresh()
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR

    @classmethod
    def check_dataset(cls):
        """Fail early, with an actionable message, when the dataset is absent."""
        missing = [p for p in (cls.IMAGE_DIR, cls.MASK_DIR) if not os.path.isdir(p)]
        if missing:
            raise FileNotFoundError(
                'Dataset directory not found:\n  ' + '\n  '.join(missing) +
                '\n\nSet Config.DATA_ROOT (or IMAGE_DIR / MASK_DIR) in config.py '
                'to point at your dataset.'
            )


# Initialise the derived paths at import time.
Config.refresh()
