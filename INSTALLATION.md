# Palette Attention Model - Fashion Generation

## Project Structure Created

```
palette-attention-model/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch Dataset + DataLoader
│   └── preprocess.py       # Edge map extraction, normalization
│
├── models/
│   ├── __init__.py
│   ├── encoder.py          # CNN encoder (ResNet-18 backbone)
│   ├── embedding.py        # Token + conditional style embeddings
│   ├── lstm_core.py        # Stacked LSTM/GRU with hidden state management
│   ├── attention.py        # Conditional spatial attention module
│   └── decoder.py          # Spatial attention decoder + output heads
│
├── configs/
│   └── default.yaml        # Hyperparameters and configuration
│
├── train.py                # Training loop, loss functions, gradient clipping
├── evaluate.py             # Evaluation, attention map visualization, FID
├── demo.py                 # Gradio interface
├── __init__.py             # Main module with complete model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Manjushwarofficial/palette-attention-model.git
cd palette-attention-model
```

2. Create conda environment:
```bash
conda create -n palette-attn python=3.9
conda activate palette-attn
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Preprocess Data
```bash
python data/preprocess.py --input data/raw/ --output data/processed/
```

### Train Model
```bash
python train.py --config configs/default.yaml
```

### Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best.pt --split val
```

### Run Demo
```bash
python demo.py
```

Then open http://localhost:7860 in your browser.

## Project Structure Overview

### Data Module (`data/`)
- **dataset.py**: PyTorch Dataset for loading fashion images and sketches
- **preprocess.py**: Edge map extraction using Canny, silhouette generation, normalization

### Models Module (`models/`)
- **encoder.py**: ResNet-18 based CNN with frozen pretrained weights
- **embedding.py**: Token embeddings with positional encoding + style embeddings
- **lstm_core.py**: Stacked LSTM/GRU layers with layer normalization
- **attention.py**: Conditional spatial attention with silhouette masking
- **decoder.py**: Spatial decoder for pixel-level reconstruction + token decoder

### Configuration (`configs/`)
- **default.yaml**: Comprehensive configuration for model, training, data, and logging

### Main Scripts
- **train.py**: Complete training loop with gradient clipping, loss tracking, checkpoint saving
- **evaluate.py**: Evaluation metrics, FID, attention visualization
- **demo.py**: Gradio web interface for interactive generation
- **__init__.py**: Main PaletteAttentionModel class orchestrating all components

## Key Features

✓ **No Transformers** — Pure RNN/LSTM architecture for sequential generation
✓ **Conditional Spatial Attention** — Per-step attention conditioned on hidden state + style
✓ **Anti-Bleeding Masking** — Silhouette-based masks prevent color leakage across boundaries
✓ **Modular Design** — Clean separation of concerns (encoder, embedding, RNN, attention, decoder)
✓ **Complete Training Pipeline** — Gradient clipping, learning rate scheduling, checkpoint management
✓ **Interactive Demo** — Gradio-based web interface for sketch-to-outfit generation
✓ **Comprehensive Evaluation** — Reconstruction loss, sequence accuracy, FID, attention visualization

## Next Steps

1. Download and prepare DeepFashion or custom dataset
2. Run preprocessing to generate edge maps and silhouettes
3. Adjust hyperparameters in `configs/default.yaml`
4. Start training: `python train.py --config configs/default.yaml`
5. Monitor progress with TensorBoard or Weights & Biases
6. Evaluate on validation set and visualize attention maps
7. Deploy with Gradio demo for interactive use

## Reference

Based on the architecture specified in the README:
- Memory-based generation using stacked RNNs/LSTMs/GRUs
- Conditional spatial attention without Transformers
- Anti-bleeding mechanism through structural masking
- Frozen ResNet-18 backbone for efficient training
