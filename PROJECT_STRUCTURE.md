"""
Project structure map and module relationships for Palette Attention Model.
"""

PROJECT_STRUCTURE = """
🎨 PALETTE ATTENTION MODEL - Complete Project Structure
═══════════════════════════════════════════════════════════════════════════════

palette-attention-model/
│
├── 📁 data/                          [Data Module - Input Processing]
│   ├── __init__.py
│   ├── dataset.py                    │ PyTorch Dataset class
│   │                                 ├─ FashionDataset: loads sketches + outfits
│   │                                 └─ get_fashion_dataloader: DataLoader factory
│   │
│   └── preprocess.py                 │ Data Preprocessing
│                                     ├─ FashionPreprocessor: main class
│                                     ├─ extract_edges(): Canny edge detection
│                                     ├─ extract_silhouette(): binary mask extraction
│                                     ├─ resize_and_normalize(): image normalization
│                                     └─ process_dataset(): batch processing
│
├── 📁 models/                        [Model Module - Architecture Components]
│   ├── __init__.py
│   │
│   ├── encoder.py                    │ CNN Encoder (Feature Extraction)
│   │                                 ├─ CNNEncoder: ResNet-18 backbone
│   │                                 ├─ Frozen pretrained weights
│   │                                 └─ Conv projection to output_channels
│   │
│   ├── embedding.py                  │ Token & Style Embeddings
│   │                                 ├─ TokenEmbedding: with positional encoding
│   │                                 ├─ StyleEmbedding: category + attribute fusion
│   │                                 └─ ConditionalEmbedding: combined module
│   │
│   ├── lstm_core.py                  │ Sequential Memory (RNN Core)
│   │                                 ├─ LSTMCore: stacked LSTM layers
│   │                                 ├─ GRUCore: alternative GRU layers
│   │                                 ├─ Layer normalization for stability
│   │                                 └─ init_hidden(): state initialization
│   │
│   ├── attention.py                  │ Conditional Spatial Attention
│   │                                 ├─ SpatialAttention: core attention module
│   │                                 ├─ Multi-step attention wrapper
│   │                                 ├─ Silhouette masking (anti-bleeding)
│   │                                 └─ Masked softmax over spatial grid
│   │
│   └── decoder.py                    │ Decoder (Output Generation)
│                                     ├─ SpatialDecoder: pixel-level reconstruction
│                                     ├─ SequentialDecoder: token prediction
│                                     ├─ CombinedDecoder: unified output heads
│                                     └─ Transposed convolutions with upsampling
│
├── 📁 configs/                       [Configuration Module]
│   └── default.yaml                  │ Hyperparameter Configuration
│                                     ├─ Model architecture settings
│                                     ├─ Training hyperparameters
│                                     ├─ Data paths and preprocessing
│                                     ├─ Loss weights
│                                     ├─ Scheduler configuration
│                                     └─ Logging and monitoring settings
│
├── 📁 notebooks/                     [Jupyter Notebooks - Analysis & Viz]
│   ├── 01_data_exploration.ipynb     │ Data loading and preprocessing
│   ├── 02_model_architecture.ipynb   │ Architecture visualization
│   ├── 03_training_analysis.ipynb    │ Loss curves and metrics
│   └── 04_results_visualization.ipynb│ Attention maps and outputs
│
├── 📄 train.py                       [Training Script]
│                                     ├─ Trainer class: orchestrates training
│                                     ├─ train_epoch(): single epoch training
│                                     ├─ validate(): validation loop
│                                     ├─ Gradient clipping (clip_grad_norm_)
│                                     ├─ Learning rate scheduling
│                                     ├─ Checkpoint management
│                                     └─ Loss tracking to Weights & Biases
│
├── 📄 evaluate.py                    [Evaluation Script]
│                                     ├─ ModelEvaluator class: metrics computation
│                                     ├─ compute_reconstruction_loss(): MSE + perceptual
│                                     ├─ compute_fid_score(): Fréchet Inception Distance
│                                     ├─ compute_sequence_accuracy(): token-level accuracy
│                                     ├─ visualize_attention_maps(): spatial visualization
│                                     └─ Full evaluation pass with all metrics
│
├── 📄 demo.py                        [Gradio Demo Interface]
│                                     ├─ PaletteAttentionDemo class
│                                     ├─ generate_outfit(): inference on sketch
│                                     ├─ Gradio UI with input/output blocks
│                                     ├─ Model loading from checkpoint
│                                     ├─ Image preprocessing/postprocessing
│                                     └─ Attention visualization in interface
│
├── 📄 __init__.py                    [Main Model Module]
│                                     ├─ PaletteAttentionModel: complete model
│                                     ├─ Orchestrates all components
│                                     ├─ forward(): combined forward pass
│                                     ├─ Attention computation and masking
│                                     └─ Multi-task output (image + tokens)
│
├── 📄 utils.py                       [Utility Functions]
│                                     ├─ GradientClipper: gradient clipping utilities
│                                     ├─ CheckpointManager: save/load checkpoints
│                                     ├─ LearningRateScheduler: custom schedulers
│                                     ├─ AverageMeter: loss tracking
│                                     ├─ set_seed(), count_parameters()
│                                     └─ freeze/unfreeze utilities
│
├── 📄 requirements.txt                [Dependencies - pip install]
├── 📄 INSTALLATION.md                 [Setup Instructions]
└── 📄 README.md                       [Project Documentation]

═══════════════════════════════════════════════════════════════════════════════

MODULE DEPENDENCY GRAPH
═══════════════════════════════════════════════════════════════════════════════

    DATA INPUTS
        │
        ├─→ data/preprocess.py ─→ Edge maps, silhouettes
        │       │
        ├─→ data/dataset.py ─→ PyTorch DataLoader
        │       │
        ▼       ▼
    train.py / evaluate.py / demo.py
        │       │                  │
        │       ├──────────────────┤
        │                          │
        ├─→ models/encoder.py (CNN) ◄──┤
        ├─→ models/embedding.py ◄──┤
        ├─→ models/lstm_core.py ◄──┤
        ├─→ models/attention.py ◄──┤
        ├─→ models/decoder.py ◄──┤
        │                          │
        └─→ __init__.py (PaletteAttentionModel)
            │
            ├─ Encoder
            ├─ Token Embedding
            ├─ Style Embedding
            ├─ LSTM/GRU Core
            ├─ Spatial Attention
            └─ Decoder
                │
                ▼
            GENERATED OUTFITS + ATTENTION MAPS


WORKFLOW FLOW
═══════════════════════════════════════════════════════════════════════════════

TRAINING FLOW:
──────────────
1. Raw Images → preprocess.py → Edge maps + Silhouettes
2. Processed Data → dataset.py → DataLoader
3. DataLoader → train.py (Trainer.train_epoch())
4. Input Sketch → encoder.py → Feature Map
5. Tokens → embedding.py → Token Embeddings
6. Features + Hidden States → attention.py → Attention Context
7. Context → decoder.py → Generated Outfit
8. Loss Computation → Backward Pass → Gradient Clipping
9. Optimizer Step → Save Checkpoint

EVALUATION FLOW:
────────────────
1. Checkpoint → evaluate.py (ModelEvaluator)
2. Generate Outputs → Compute Metrics
3. Metrics: MSE, Perceptual Loss, FID, Sequence Accuracy
4. Visualize Attention Maps → Output Images

INFERENCE FLOW:
────────────────
1. User Upload Sketch → demo.py
2. Preprocess → encoder.py
3. Generate → __init__.py (PaletteAttentionModel)
4. Postprocess → Display Results + Attention Maps


KEY ARCHITECTURE DECISIONS
═══════════════════════════════════════════════════════════════════════════════

✓ NO TRANSFORMERS: Pure LSTM/GRU for sequential memory
✓ CONDITIONAL ATTENTION: Per-step, conditioned on hidden state + style
✓ SILHOUETTE MASKING: Prevents color bleeding across boundaries
✓ FROZEN BACKBONE: ResNet-18 not trained, only downstream layers
✓ SPATIAL CONTEXT: Features extracted as spatial maps, not flattened
✓ MODULAR DESIGN: Each component independently reusable
✓ MULTI-TASK OUTPUT: Both pixel-level and token-level predictions
✓ GRADIENT CLIPPING: Mandatory for LSTM stability


FILE LOCATIONS & PURPOSES
═══════════════════════════════════════════════════════════════════════════════

Code Structure:
├── data/                  - Input pipeline
├── models/                - Architecture modules
├── configs/               - Configuration parameters
├── notebooks/             - Analysis & visualization
└── Root level scripts     - Entry points

Data Structure (after preprocessing):
├── data/
│   ├── raw/               - Original images (user provides)
│   └── processed/
│       ├── images/        - Normalized outfit images
│       ├── edges/         - Edge maps from Canny
│       └── silhouettes/   - Binary garment masks

Output Structure:
├── checkpoints/           - Saved model weights
├── outputs/               - Generated images + visualizations
└── logs/                  - TensorBoard logs, etc.

"""

print(PROJECT_STRUCTURE)
