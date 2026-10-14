# BCM-UNet: Boundary-Guided Crack Segmentation and Quantification for Coal CT Images

Reference implementation of a boundary-guided multi-scale U-Net (BCM-UNet) for
segmenting fractures in coal CT images, together with a skeleton-based
quantification pipeline and a desktop application (CrackQuant).

The model combines three modules, each of which can be switched on or off
independently for ablation studies:

| Switch | Module | Purpose |
| --- | --- | --- |
| `USE_BOUNDARY_MODULE` | Boundary module | Predicts a high-resolution edge map and uses it to gate every decoder stage. The edge prediction and the joint edge + segmentation loss form one module, so this single switch controls the architecture and its loss together. |
| `USE_CONTINUITY_MODULE` | Continuity enhancement | Dilated multi-receptive-field block that reconnects broken cracks |
| `USE_MULTISCALE_FUSION` | Multi-scale fusion decoder | Fuses decoder features at several resolutions for uneven crack widths |

Each switch removes its module, and its parameters, completely when disabled.
The run name
`SR{B}{C}{M}` records which modules are active, where `B`, `C` and `M` correspond
to the three switches above. With all three off, `BoundaryUNet` reduces to a
plain U-Net decoder.

## Repository layout

```
config.py                    Central configuration and ablation control panel
train.py                     Training entry point
dataset.py                   Paired image/mask dataset
predict.py                   Single-image inference
models/                      UNet baseline and BoundaryUNet (BCM-UNet)
utils/                       Loss functions and visualisation helpers
CrackQuantGUI_English/       PyQt5 desktop application
```

## Installation

```bash
git clone https://github.com/nhyzc/BCM-UNet.git
cd BCM-UNet
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training. Inference also runs on CPU.

The results in the paper were produced with PyTorch 2.10.0 and CUDA 12.8 on
Windows 10 22H2, an Intel Core i5-14600K CPU, 64 GB of memory and an NVIDIA
GeForce RTX 5070 GPU with 12 GB of memory.

## Dataset

The dataset is **not** distributed with this repository. The training code
expects a directory containing paired images and masks:

```
augmented/
  images/     *.png   input CT slices
  labels/     *.jpg   binary crack masks
```

By default this is looked up as a sibling of the repository root. Point
`Config.DATA_ROOT` in `config.py` at your own copy if it lives elsewhere.

## Training

Edit `config.py` — it is the only file you need to change to select a variant:

```python
MODEL_NAME             = 'BoundaryUNet'   # or 'UNet' for the baseline
USE_BOUNDARY_MODULE    = True
USE_CONTINUITY_MODULE  = True
USE_MULTISCALE_FUSION  = True
```

Then run:

```bash
python train.py
```

Checkpoints, the CSV training log and periodic validation figures are written
to `output_results/<run name>/`. The run name is derived from the switches
(`BoundaryUNet_SR111` with all three modules enabled, `UNet_Baseline` for the
baseline). To train another variant, change the switches and run
`python train.py` again.

## Inference

```bash
python predict.py --image path/to/slice.png
```

The script loads the checkpoint selected in `config.py`; pass `--model` to point
at a different one. Results are written to `<run dir>/predictions/`.

## Desktop application

`CrackQuantGUI_English/` is a PyQt5 application that wraps segmentation and
quantification into a single interface: open an image, run segmentation, run
quantification, inspect the crack table (ID, length, width, inclination) and
export the results.

```bash
cd CrackQuantGUI_English
python main.py
```

To build a standalone executable:

```bash
cd CrackQuantGUI_English
python build_exe.py
```

## Code availability

The BCM-UNet implementation, the fracture quantification pipeline and the
CrackQuant desktop application are distributed in this repository. The datasets
used in the study are publicly available from the Digital Porous Media Portal
(DRP-20_archive and DRP-21_archive); the fracture annotations are available
from the corresponding author on reasonable request.

## Quantification pipeline

`CrackQuantGUI_English/quantify.py` implements the measurement stage:

1. Zhang-Suen skeletonisation, endpoint extension and junction repair;
2. classification of endpoints and junction nodes;
3. branch separation by tracing with direction continuity at junctions;
4. per-branch trace length, mean width (from the Euclidean distance transform,
   excluding junction neighbourhoods) and inclination.

## Citation

If you use this software, please cite the article:

```bibtex
@article{Yu2026BCMUNet,
  title   = {Intelligent segmentation and structural quantification of
             fractures in coal CT images},
  author  = {Yu, Hao and Wang, Chao and He, Ziwang and Ying, Junchao and
             Yang, Siyuan and Jin, Zijun},
  journal = {Computers \& Geosciences},
  year    = {2026}
}
```

Citation metadata is also provided in [CITATION.cff](CITATION.cff).

## License

Released under the MIT License. See [LICENSE](LICENSE).
