# Intel Image Classification CNN

A PyTorch project that applies **transfer learning** with a pretrained **ResNet-50** model to classify natural scene images from the **Intel Image Classification** dataset. The project combines modern data augmentation, progressive layer unfreezing, mixed precision training, and learning rate scheduling to improve model performance while reducing training time.

## Overview

Rather than training a CNN from scratch, this project fine-tunes a pretrained **ResNet-50** model on the Intel Image Classification dataset. Training begins with the classifier head while the backbone remains frozen, followed by progressively unfreezing deeper layers to adapt the pretrained features to the new task.

The training pipeline also incorporates several optimization techniques commonly used in modern computer vision workflows.

## Features

- Transfer learning with pretrained **ResNet-50**
- Progressive layer unfreezing
- Mixed precision training (AMP)
- AdamW optimizer
- Cosine Annealing learning rate scheduler
- Label smoothing for improved generalization
- Advanced image augmentation
- Experiment tracking with Weights & Biases (W&B)
- GPU (CUDA) support

## Project Structure

```text
.
├── model.ipynb    # Complete training and evaluation pipeline
└── README.md
```

## Dataset

The project uses the **Intel Image Classification** dataset containing six natural scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Images are loaded using PyTorch's `ImageFolder` dataset and organized into training and testing directories.

## Training Pipeline

The project follows a staged transfer learning approach:

### Stage 1 – Feature Extraction

- Freeze the pretrained ResNet-50 backbone
- Train only the final classification layer
- AdamW optimizer (`lr = 1e-3`)

### Stage 2 – Progressive Unfreezing

- **Epoch 6:** Unfreeze `layer4`
- **Epoch 11:** Unfreeze the entire network
- Lower learning rates used during fine-tuning

### Optimization

- **Optimizer:** AdamW
- **Loss:** CrossEntropyLoss with label smoothing
- **Scheduler:** Cosine Annealing LR
- **Epochs:** 15
- **Mixed Precision:** Automatic Mixed Precision (AMP)

## Data Augmentation

Training images undergo several augmentation techniques, including:

- Random resized cropping
- Horizontal flipping
- Color jitter
- Random perspective transformation
- Random erasing
- Image normalization using ImageNet statistics

Validation images are resized, center-cropped, and normalized using the pretrained ResNet-50 transforms.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/shehryar11w/Intel-Image-classification-CNN.git
cd Intel-Image-classification-CNN
```

Install the required dependencies:

```bash
pip install torch torchvision matplotlib wandb
```

Open `model.ipynb` in Jupyter Notebook or VS Code and run the notebook to train and evaluate the model.

## Technologies

- Python
- PyTorch
- TorchVision
- Matplotlib
- Weights & Biases (W&B)

## Learning Outcomes

This project demonstrates practical techniques for training high-performing image classification models, including:

- Transfer learning with pretrained CNNs
- Progressive layer unfreezing
- Modern data augmentation strategies
- Learning rate scheduling
- Mixed precision training
- Experiment tracking and model optimization

## License

This project is intended for educational purposes.
