# Pet Dataset CNN

A PyTorch project exploring **transfer learning** for image classification on the **Oxford-IIIT Pet Dataset**. The repository compares three common training strategies using a pretrained **ResNet-50** model to demonstrate the trade-offs between training speed, computational cost, and model performance.

## Overview

Instead of training a convolutional neural network from scratch, this project leverages a pretrained ResNet-50 and evaluates different fine-tuning approaches:

1. **Feature Extraction** – Freeze the pretrained backbone and train only the classification head.
2. **Full Fine-Tuning** – Update all model parameters using a small learning rate.
3. **Progressive Unfreezing** – Gradually unfreeze layers during training to improve adaptation while retaining pretrained knowledge.

All experiments are implemented in **PyTorch** and trained on the **Oxford-IIIT Pet Dataset**, which contains **37 cat and dog breeds**.

## Features

- Transfer learning with pretrained **ResNet-50**
- Three training strategies for comparison
- Automatic dataset download using `torchvision`
- Image preprocessing using pretrained model transforms
- 80/20 train-validation split
- AdamW optimizer
- Cross-Entropy loss
- Experiment tracking with Weights & Biases (W&B)
- GPU (CUDA) support when available

## Project Structure

```text
.
├── model-feature-extraction.ipynb
│   └── Freeze ResNet backbone and train only the classifier

├── model-fine-tuning.ipynb
│   └── Fine-tune the entire pretrained network

├── model-progressive-unfreezing.ipynb
│   └── Gradually unfreeze layers during training

└── README.md
```

## Dataset

The project uses the **Oxford-IIIT Pet Dataset** provided by `torchvision`.

- **37** pet categories
- Images automatically downloaded on first run
- Images resized and normalized using the pretrained ResNet-50 transforms

## Model Configurations

### Feature Extraction

- Frozen ResNet-50 backbone
- Train only the final fully connected layer
- AdamW optimizer (`lr = 1e-3`)
- 15 training epochs

### Full Fine-Tuning

- Entire ResNet-50 is trainable
- AdamW optimizer (`lr = 1e-5`)
- 15 training epochs

### Progressive Unfreezing

- Initially freeze the backbone
- Gradually unfreeze deeper layers during training
- AdamW optimizer
- 15 training epochs

## Getting Started

Clone the repository:

```bash
git clone https://github.com/shehryar11w/petDataset-CNN.git
cd petDataset-CNN
```

Install the dependencies:

```bash
pip install torch torchvision matplotlib wandb
```

Run any notebook using Jupyter Notebook or VS Code.

The dataset will be downloaded automatically the first time the notebooks are executed.

## Technologies

- Python
- PyTorch
- TorchVision
- Matplotlib
- Weights & Biases (W&B)

## Learning Outcomes

This project explores practical transfer learning techniques and demonstrates:

- Using pretrained CNNs for image classification
- Feature extraction vs. fine-tuning
- Progressive layer unfreezing
- Transfer learning best practices
- Training and evaluating deep learning models in PyTorch

## License

This project is intended for educational purposes.
