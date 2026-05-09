# Deep Learning Image Colorization

A convolutional neural network pipeline for automatic colorization of grayscale images. The model is trained on face images and then fine-tuned on a fruits & vegetables dataset using transfer learning. Built with PyTorch and OpenCV.

## Overview

Colorization is framed as a regression problem in the LAB color space. The network takes a grayscale **L channel** as input and predicts the **a\* and b\* chrominance channels**, which are then combined with L to reconstruct a full-color image.

The pipeline consists of two models:

- **Regressor** — a lightweight CNN that predicts the mean a\* and b\* values of an image from its L channel. Used as a baseline to understand chrominance distribution.
- **Colorizer** — a deeper encoder-decoder CNN (similar in spirit to a U-Net without skip connections) that produces spatially-aware colorization at 128×128 resolution.

### Architecture

```
Input: (1, 128, 128)  →  Encoder  →  (512, 4, 4)  →  Decoder  →  Output: (2, 128, 128)
```

**Encoder:** 5 strided Conv2d layers (1→64→128→256→512→512) with BatchNorm + ReLU  
**Decoder:** 5 Upsample + Conv2d layers (512→512→256→128→64→2) with BatchNorm + ReLU + final Tanh

### Training Details

| Stage | Dataset | Epochs | Augmentation |
|---|---|---|---|
| Initial training | Face images (90/10 split) | 100 | Random flip, random crop, brightness jitter (10× augmentation) |
| Transfer learning | NCD fruits & vegetables | 50 | None (encoder frozen) |

- **Loss:** MSE on predicted vs. ground truth a\*, b\* channels
- **Optimizer:** Adam (lr=0.001 for training, lr=0.0001 for fine-tuning)
- **Batch size:** 10

## Project Structure

```
.
├── colorizer.py        # Colorizer model definition
├── regressor.py        # Regressor model definition + training script
├── train.py            # Train the Colorizer on face images
├── test.py             # Evaluate on test set, save colorized outputs
├── transfer.py         # Fine-tune on NCD dataset via transfer learning
├── gpu_speedup.py      # Benchmark GPU vs CPU inference time
├── face_images/        # Training dataset (grayscale face images)
├── NCDataset/          # NCD grayscale images (transfer learning input)
└── NCColorful/         # NCD color images (transfer learning targets)
```

> **Note:** Model weights (`*.pth`) are not included. Run the scripts in order to generate them.

## Requirements

```bash
pip install torch torchvision opencv-python numpy
```

A CUDA-capable GPU is recommended. The scripts default to `device = cuda`.

## How to Run

Run the scripts in the following order:

### 1. Train the Regressor (baseline)

```bash
python3 regressor.py
```

Trains a small CNN to predict mean chrominance (a\*, b\*) from the L channel. Outputs augmented images and LAB channel visualizations to `augmented/`, `L/`, `a/`, and `b/`.

### 2. Train the Colorizer

```bash
python3 train.py
```

Trains the full encoder-decoder Colorizer on the face image dataset for 100 epochs with 10× data augmentation. Saves weights to `colorizer.pth`.

### 3. Evaluate on Test Set

```bash
python3 test.py
```

Loads `colorizer.pth`, runs inference on the held-out test faces, and saves colorized results to `colorized_results/`. Also saves 10 colorized training examples to `colorized_train_results/` and reports per-channel output variance.

### 4. Fine-tune via Transfer Learning

```bash
python3 transfer.py
```

Loads the pretrained `colorizer.pth`, freezes the encoder, and fine-tunes only the decoder on the NCD fruits & vegetables dataset for 50 epochs. Saves colorized NCD results to `ncd_results/`.

### 5. Benchmark GPU vs CPU Speedup

```bash
python3 gpu_speedup.py
```

Runs inference on 100 face images on both CPU and GPU and reports wall-clock time and speedup factor.

## Results

| Evaluation | MSE |
|---|---|
| Faces test set | 0.000211 |
| NCD test set (after transfer) | *(printed at runtime)* |
