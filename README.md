# Sky Replacement Project

Full end-to-end pipeline for automatic sky replacement on still images. The workflow combines human matting, deep-learning sky segmentation (via NCNN), classical color fallbacks, mask refinement, and final compositing to output photo-realistic sky swaps.

## Features
- Human foreground segmentation using Robust Video Matting (TorchHub).
- Sky detection with a lightweight NCNN model and automatic color-based fallback.
- Native Windows mask refinement through `run/mask_refine.exe`.
- Optional bitwise post-processing for crisp binary masks.
- Sky compositing with configurable blending, feathering, and sky image selection.

## Repository Layout
- `scripts/sky_replacement.py` – Master pipeline that generates masks and produces final composites.
- `scripts/sky_segmentation_ncnn_refined_mask.py` – Standalone sky-mask generator with NCNN + optional refinement.
- `models/` – NCNN sky segmentation weights (`*.param`, `*.bin`).
- `run/mask_refine.exe` – Closed-source executable used to refine masks (Windows only).
- `requirements.txt` – Python dependencies.
- Expected runtime folders (create as needed):
  - `data/raw/` – Input images.
  - `data/sky_masks/` – Generated intermediate masks.
  - `data/skies/sky.png` – Default replacement sky (supply your own).
  - `outputs/final_test/` – Final composites written by the main pipeline.

## Prerequisites
- Windows 10/11 (pipeline uses a Windows `.exe` for refinement; rest works cross-platform if you disable that step).
- Python 3.10 or newer.
- (Optional) CUDA-capable GPU for faster Torch inference.
- (Optional) [Tencent NCNN](https://github.com/Tencent/ncnn) Python package; the scripts fall back to color heuristics if it is unavailable.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The `ncnn` Python wheel is only required if you plan to run NCNN-based segmentation. Skip it (or uninstall after installing requirements) if no compatible wheel is available for your platform—the scripts will switch to color-based detection and continue running.

## Preparing Assets
1. Place your input photos in `data/raw/`.
2. Copy the provided NCNN weights into `models/` (already included in this repo).
3. Add `mask_refine.exe` to `run/` (already included) and ensure all required DLLs reside beside it.
4. Supply a replacement sky image at `data/skies/sky.png`. You can swap this per run by passing `sky_image_path` to the pipeline.

## Usage

### Full Pipeline (recommended)
Runs human segmentation, sky detection, mask refinement, and final compositing.

```bash
python scripts/sky_replacement.py
```

Outputs:
- Human masks, intermediate sky masks, and final combined masks in `data/sky_masks/`.
- Composited images in `outputs/final_test/`.

Key toggles inside the script:
- Set `NCNN_AVAILABLE = False` to force color fallback.
- Adjust `feather_strength`, `alpha_blend`, or `mask_soften` within `replace_sky_using_final_mask()` for different blend characteristics.

### Sky Segmentation Only
Generates sky masks for a directory of images; optionally refines masks and saves diagnostic overlays.

```bash
python scripts/sky_segmentation_ncnn_refined_mask.py \
  --input data/raw \
  --output data/masks \
  --visualize \
  --min-sky 5.0
```

Flags:
- `--no-refinement` – Skip `mask_refine.exe`.
- `--visualize` – Store side-by-side overlays beside each mask.
- `--min-sky` – Ignore images with sky coverage below the given percentage.

## Customising the Pipeline
- **Different sky assets:** Provide a path via `sky_image_path` when calling `replace_sky_using_final_mask()` or swap files in `data/skies/`.
- **Refinement on non-Windows systems:** Set `use_refinement=False` (or delete/rename the `.exe`) and rely on color/NCNN mask outputs.
- **Batch location:** Change `RAW_DIR`, `SKY_MASK_DIR`, etc. at the top of `scripts/sky_replacement.py` if the project lives outside `F:\Studio\sky_replacement_project`.

## Troubleshooting
- *TorchHub download stalls* – Pre-download the RVM model and point TorchHub cache to a local path via the `TORCH_HOME` environment variable.
- *`ncnn` import fails* – Remove it from `requirements.txt` or install from source following the official instructions. The code will default to color segmentation.
- *`mask_refine.exe` cannot find DLLs* – Copy the required DLLs into `run/` or add their location to your `PATH`.
- *Outputs look aliased* – Increase `feather_strength` or enable additional Gaussian blur within `replace_sky_using_final_mask()`.

## Credits
- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) by Peter Lin.
- Sky segmentation model adapted from [xiongzhu666/Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing).
- Mask refinement executable courtesy of the original SkyAR toolkit.


