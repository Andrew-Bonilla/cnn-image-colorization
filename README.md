# Deep Learning Image Colorization

A convolutional neural network pipeline for automatic colorization of grayscale images. The model is trained on face images and then fine-tuned on a fruits & vegetables dataset using transfer learning. Built with PyTorch and OpenCV.

For a detailed write-up of the methodology and results, see [Formal-Writeup.pdf](Formal-Writeup.pdf).

## Overview

Colorization is framed as a regression problem in the LAB color space. The network takes a grayscale **L\* channel** as input and predicts the **a\* and b\* chrominance channels**, which are then combined with L\* to reconstruct a full-color image.

The pipeline consists of two models:

- **Regressor** — a lightweight CNN that predicts the mean a\* and b\* values of an image from its L\* channel. Used as a baseline to understand chrominance distribution.
- **Colorizer** — a deeper encoder-decoder CNN that produces spatially-aware colorization at 128×128 resolution.

### Regressor Architecture

A 7-layer CNN where each layer halves the spatial dimensions via stride-2 convolutions (128→64→32→16→8→4→2→1). All hidden layers use 3 feature maps. A final fully connected layer maps the 3-dimensional output to 2 scalars representing mean a\* and mean b\*.

### Colorizer Architecture

```
Input: (1, 128, 128)  →  Encoder  →  (512, 4, 4)  →  Decoder  →  Output: (2, 128, 128)
```

**Encoder:** 5 strided Conv2d layers (1→64→128→256→512→512) with BatchNorm + ReLU  
**Decoder:** 5 Upsample + Conv2d layers (512→512→256→128→64→2) with BatchNorm + ReLU + final Tanh

The final layer uses **Tanh activation** (following Iizuka et al., 2016), with a\* and b\* targets rescaled to [-1, 1] during training and returned to [0, 1] during inference.

### Training Details

| Stage | Dataset | Epochs | Augmentation |
|---|---|---|---|
| Initial training | Face images (90/10 split) | 100 | Random flip, random crop, brightness jitter (10× augmentation) |
| Transfer learning | NCD fruits & vegetables (721 images, 20 categories) | 50 | None (encoder frozen) |

- **Loss:** MSE on predicted vs. ground truth a\*, b\* channels
- **Optimizer:** Adam (lr=0.001 for training, lr=0.0001 for fine-tuning)
- **Batch size:** 10
- **Reproducibility:** `torch.manual_seed(42)` is used in `train.py` and `test.py` to ensure a consistent train/test split

## Hyperparameter Tuning

Three encoder/decoder variants were evaluated on the faces test set:

| Architecture | Encoder Channels | Test MSE |
|---|---|---|
| Small | 1→32→64→128→256→256 | 0.000352 |
| **Medium** | **1→64→128→256→512→512** | **0.000192** |
| Large | 1→128→256→512→512→512 | 0.000221 |

The **Medium architecture** achieved the best generalization. The Large model likely overfit given the small dataset size, and the Small model — despite the lowest training loss (0.000154) — had the highest test MSE, a clear sign of underfitting. The Medium architecture was selected for the transfer learning stage.

## Results

| Evaluation | MSE |
|---|---|
| Faces test set | 0.000211 |
| NCD test set (after transfer learning) | 0.010744 |

The higher NCD MSE reflects the greater difficulty of colorizing diverse natural objects compared to faces. Visual inspection showed reasonable fruit-appropriate colors, though a slight bias toward the face training domain was visible in some outputs.

### Tanh vs. No Tanh

| Model | Test MSE |
|---|---|
| Without Tanh | 0.000211 |
| With Tanh | 0.000313 |

### GPU Speedup

Inference on 100 face images:

| Device | Time |
|---|---|
| CPU | 0.3975s |
| GPU (NVIDIA T4) | 0.0610s |
| **Speedup** | **6.52x** |

## Project Structure

```
.
├── colorizer.py        # Colorizer model definition
├── regressor.py        # Regressor model definition + training script
├── train.py            # Train the Colorizer on face images
├── test.py             # Evaluate on test set, save colorized outputs
├── transfer.py         # Fine-tune on NCD dataset via transfer learning
└── gpu_speedup.py      # Benchmark GPU vs CPU inference time
```

> **Datasets:** Not included in this repo. Place your data in the following directories before running:
> - `face_images/` — 128×128 face images used for initial training ([Georgia Tech Face Database](https://www.anefian.com/research/face_reco.htm))
> - `NCDataset/` — NCD grayscale images (transfer learning input) ([Natural Color Dataset](https://github.com/saeed-anwar/ColorSurvey), [paper](https://arxiv.org/pdf/2008.10774))
> - `NCColorful/` — NCD color images (transfer learning targets) ([Natural Color Dataset](https://github.com/saeed-anwar/ColorSurvey), [paper](https://arxiv.org/pdf/2008.10774))
>
> **Model weights:** `colorizer.pth` is included. You can also regenerate it by running the scripts in order below.

## Requirements

```bash
pip install torch torchvision opencv-python numpy
```

A CUDA-capable GPU is recommended. The scripts default to `device = cuda`.

## How to Run

Run the scripts in the following order. **If you want to use the provided `colorizer.pth` weights, you can skip steps 1 and 2 and start from step 3.**

### 1. Train the Regressor (baseline)

```bash
python3 regressor.py
```

Trains a small CNN to predict mean chrominance (a\*, b\*) from the L\* channel. Outputs augmented images and LAB channel visualizations to `augmented/`, `L/`, `a/`, and `b/`.

### 2. Train the Colorizer

```bash
python3 train.py
```

Trains the encoder-decoder Colorizer on the face image dataset for 100 epochs with 10× data augmentation. Saves weights to `colorizer.pth`.

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
