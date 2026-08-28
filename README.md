# Adaptive Contrast Restoration for Low-Dose Chest Radiography

This project investigates acquisition degradation and spatial-domain
enhancement of chest radiographs using MATLAB.

## Dataset

A working subset of 300 frontal chest radiographs was selected from the
NIH ChestX-ray14 dataset.

Dataset:
NIH Clinical Center ChestX-ray14

The dataset itself is not included in this repository.

## Phase 1 - Acquisition Modelling

- Spatial resolution degradation
  - Downsampling factors: 2, 4, 8
  - Nearest-neighbour
  - Bilinear
  - Bicubic

- Intensity quantization
  - 7, 6, 5, 4, 3 and 2 bit

- Low-dose simulation
  - 50%
  - 25%
  - 10%
  - Poisson photon-counting noise

Metrics:
- MSE
- RMSE
- PSNR
- SSIM
- SNR

## Phase 2 - Enhancement

Methods evaluated:
- Histogram equalization
- Histogram matching
- AHE
- Custom CLAHE
- Box filtering
- Gaussian filtering
- Median filtering
- Adaptive median filtering
- Alpha-trimmed mean filtering
- Laplacian sharpening
- Unsharp masking
- High-boost filtering

Selected enhancement chain:

Gaussian smoothing
→ Histogram matching
→ High-boost filtering

## Phase 3 - Evaluation

Evaluation includes:

- PSNR
- SSIM
- RMSE
- Entropy
- EME
- Local contrast
- CLAHE sensitivity analysis
- Visual preference study

## Software

MATLAB / MATLAB Online

## Dataset citation

Wang et al., ChestX-ray8/ChestX-ray14,
CVPR 2017, NIH Clinical Center.
