# Deep Learning & LLMs

A collection of deep learning projects and exercises spanning neural networks built from scratch, computer vision, sequence models, and transformer/LLM fundamentals — built for hands-on learning of the concepts underlying modern deep learning.

## Projects

Every project below lives in its own repository and is linked into this repo as a **git submodule**.

Every sub-project has its own `README.md` with full setup/training details — the table below is a quick summary.

| Project | Repository | Description |
|---|---|---|
| [`BasicNN-manual+TF`](./BasicNN-manual+TF) | [`NN-basics`](https://github.com/shehryar11w/NN-basics) | A basic neural network implemented manually (pure Python/NumPy) alongside the same network built in TensorFlow, for comparing what the framework abstracts away. |
| [`CNN basics`](./CNN%20basics) | [`CNN-basics`](https://github.com/shehryar11w/CNN-basics) | Core CNN concepts, including a manual ResNet implementation and a ResNet with training optimizations (learning rate scheduling, etc.). |
| [`Intel-Image`](./Intel-Image) | [`Intel-Image-classification-CNN`](https://github.com/shehryar11w/Intel-Image-classification-CNN) | Transfer learning with a pretrained ResNet-50 on the Intel Image Classification dataset. Covers progressive layer unfreezing, mixed precision training, AdamW, cosine annealing LR, and W&B experiment tracking. |
| [`MicroGrad`](./MicroGrad) | [`MicroGrad`](https://github.com/shehryar11w/MicroGrad) | A minimal, from-scratch autograd engine and neural net library (`Value`, `Neuron`, `Layer`, `MLP`) — implementing backpropagation from first principles, in the spirit of Karpathy's micrograd. |
| [`PetDataset`](./PetDataset) | [`petDataset-CNN`](https://github.com/shehryar11w/petDataset-CNN) | Transfer learning on the Oxford-IIIT Pet Dataset (37 breeds) with ResNet-50, comparing three strategies side by side: feature extraction, full fine-tuning, and progressive unfreezing. |
| [`Pascal-VOC`](./Pascal-VOC) | [`Pascal-VOC`](https://github.com/shehryar11w/Pascal-VOC) | Object detection on Pascal VOC 2012 (20 classes), comparing a two-stage Faster R-CNN against a one-stage YOLOv8 model. |
| [`RNN-basics`](./RNN-basics) | [`RNN-basics`](https://github.com/shehryar11w/RNN-basics) | Vanilla RNN, LSTM, and GRU cells implemented manually for character-level language modeling on tiny-Shakespeare, with a head-to-head comparison. |
| [`Seq2Seq`](./Seq2Seq) | [`Seq2Seq`](https://github.com/shehryar11w/Seq2Seq) | English-to-French translation with an encoder-decoder model, implemented both with and without attention. |
| [`ViT`](./ViT) | [`ViT`](https://github.com/shehryar11w/ViT) | A Vision Transformer built from scratch (patch embeddings, CLS token, multi-head self-attention, transformer encoder), compared against a pretrained ViT. |
| [`YOLO-fundamentals`](./YOLO-fundamentals) | [`YOLO-fundamentals`](https://github.com/shehryar11w/YOLO-fundamentals) | Core detection building blocks implemented from scratch: IoU and Non-Maximum Suppression. |
| [`carvana-image_segmentation`](./carvana-image_segmentation) | [`carvana-image_segmentation`](https://github.com/shehryar11w/carvana-image_segmentation) | A manually-built U-Net for semantic segmentation, trained on Oxford-IIIT Pet trimap masks with a combined Cross-Entropy + Dice loss. |
| [`manualEmbedding`](./manualEmbedding) | [`manualEmbedding`](https://github.com/shehryar11w/manualEmbedding) | Word2Vec (Skip-Gram with negative sampling) implemented from scratch, then used for IMDB sentiment classification and compared against pretrained GloVe embeddings. |
| [`PyTorch Basics`](./PyTorch%20Basics) | *(not a submodule)* | Fundamentals of PyTorch — tensors, autograd, and the basic training loop. |

## Tech Stack

- **Python**
- **PyTorch** / **TorchVision**
- **TensorFlow** (in `BasicNN-manual+TF`)
- **NumPy**
- **Matplotlib**
- **Weights & Biases (W&B)** for experiment tracking
- Jupyter Notebooks for most experiments

## Getting Started

This repo uses **git submodules**, so clone it with `--recurse-submodules` to pull in every sub-project:

```bash
git clone --recurse-submodules https://github.com/shehryar11w/Deep-Learning-and-LLMs.git
cd Deep-Learning-and-LLMs
pip install torch torchvision tensorflow numpy matplotlib wandb jupyter
```

If you already cloned without that flag, pull the submodules in afterward:

```bash
git submodule update --init --recursive
```

Each project is its own repository, linked in as a submodule, and is self-contained — open its notebook(s) in Jupyter or VS Code and run. Some sub-projects have their own `README.md` with more specific setup and training details.

To pull in the latest changes from each sub-project's repo later on:

```bash
git submodule update --remote --merge
```

## Purpose

This repo is a personal learning log for deep learning and LLMs — moving from manual, from-scratch implementations of core building blocks (autograd, basic NNs, embeddings) up to modern architectures and training techniques (CNNs, transfer learning, ViT, sequence models, object detection/segmentation).

## License

Educational / personal use.
