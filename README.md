# Text-to-Image Generation from Scratch

A minimal, end-to-end implementation of a text-to-image generative model built entirely from scratch using PyTorch. This project demonstrates **compositional generalization** — the ability to generate unseen combinations of learned concepts (e.g., generating a "maroon circle" after training on "red circle" and "maroon square").

**Dataset**: [Colored Geometrics on Kaggle](https://www.kaggle.com/datasets/mahamedmahmoud/colored-geometrics)

## Key Feature: Compositional Generalization

Unlike naive approaches that memorize exact (image, caption) pairs, this implementation:
- Uses **zero duplicate images** — each (shape, color) combination appears exactly once
- Splits data **compositionally**: training sees some colors for each shape, validation tests **unseen colors** for the same shapes
- Tests whether the model learns "red", "circle", "maroon" as **separate composable concepts**

## Repository Structure

```
├── 1_generate_dataset.py      # Generate 2,000+ unique shape-color combinations
├── 2_train_model.py           # Train with compositional validation split
├── 3_generate_image.py        # Generate + test compositional generalization
├── generate.bat               # Windows launcher
├── generate.ps1               # PowerShell launcher
├── README.md                  # This file
├── shapes_dataset/            # Generated dataset (or download from Kaggle)
│   ├── train/
│   │   └── images/            # Training images (seen combinations)
│   ├── val/
│   │   └── images/            # Validation images (unseen combinations)
│   ├── train_captions.json
│   ├── val_captions.json
│   ├── vocab.json
│   └── split_info.json        # Documents held-out colors per shape
└── checkpoints/               # Saved model weights (after training)
    ├── best_model.pt          # Best validation performance
    ├── final_model.pt
    ├── checkpoint_epoch{10,20,...}.pt
    ├── loss_curve.png
    └── visualizations/
        └── epoch_001.png      # Sample outputs per epoch
```

## Dataset

Due to GitHub file size limitations, the dataset is hosted on Kaggle:

**[Colored Geometrics Dataset](https://www.kaggle.com/datasets/mahamedmahmoud/colored-geometrics)**

The dataset contains:
- **2,220 unique images** (22 shapes × 101 colors)
- **Zero duplicates** — each (shape, color) pair appears exactly once
- **Compositional split**: 80% train (seen combinations), 20% val (unseen combinations)
- **Image format**: 64×64 RGB PNG, single shape filling entire canvas

### Shapes (22)
circle, square, triangle, rectangle, ellipse, pentagon, hexagon, heptagon, octagon, nonagon, decagon, star, heart, diamond, semicircle, oval, cross, arrow, moon, ring, trapezoid, parallelogram

### Colors (101)
Standard colors (red, blue, green, yellow, purple, orange, pink, cyan, brown, white, black, gray) plus 89 extended colors including maroon, olive, teal, navy, lime, indigo, violet, magenta, crimson, coral, salmon, gold, khaki, beige, ivory, plum, orchid, tan, peru, sienna, chocolate, turquoise, aquamarine, skyblue, steelblue, royalblue, slateblue, forestgreen, seagreen, springgreen, chartreuse, lawngreen, yellowgreen, firebrick, tomato, orangered, darkorange, goldenrod, darkgoldenrod, rosybrown, saddlebrown, darkgreen, darkcyan, darkblue, midnightblue, darkslategray, dimgray, slategray, lightslategray, lightsteelblue, powderblue, paleturquoise, lightcyan, aliceblue, ghostwhite, lavender, thistle, mistyrose, antiquewhite, linen, oldlace, floralwhite, cornsilk, lemonchiffon, lightyellow, honeydew, mintcream, azure, snow, seashell, peachpuff, bisque, moccasin, navajowhite, wheat, burlywood, darkkhaki, palegreen, lightgreen, mediumspringgreen, mediumseagreen, mediumaquamarine, cadetblue, cornflowerblue, deepskyblue, dodgerblue, lightseagreen, mediumturquoise, hotpink, deeppink, palevioletred, mediumvioletred, palegoldenrod, mediumorchid, mediumpurple, rebeccapurple, blueviolet, darkorchid, darkviolet

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- Pillow, NumPy, Matplotlib

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Quick Start

### Option 1: Use Pre-generated Dataset (Recommended)

Download from Kaggle and extract to `./shapes_dataset/`:

```bash
# Download from: https://www.kaggle.com/datasets/mahamedmahmoud/colored-geometrics
# Extract to your project folder
```

### Option 2: Generate Dataset from Scratch

```bash
python 1_generate_dataset.py --output ./shapes_dataset --size 64
```

This creates 2,000+ unique images with compositional train/val split.

### Step 2: Train the Model

```bash
python 2_train_model.py --data_dir ./shapes_dataset --epochs 100 --patience 20
```

**Training features:**
- Loads pre-split train/val folders
- Validates on **unseen** (shape, color) combinations
- Early stopping with patience=20
- Saves best model based on validation loss
- Generates visualization grids each epoch

### Step 3: Generate Images

```bash
# Basic generation (seen combination)
python 3_generate_image.py --prompt "a red circle" --model ./checkpoints/best_model.pt

# Test compositional generalization (unseen combination)
python 3_generate_image.py --prompt "a maroon circle" --model ./checkpoints/best_model.pt

# Batch test on held-out validation combinations
python 3_generate_image.py --test_compositional --data_dir ./shapes_dataset --model ./checkpoints/best_model.pt

# Compare best vs final model
python 3_generate_image.py --compare --prompt "a teal square"

# Grid of multiple prompts
python 3_generate_image.py --prompts "red circle" "maroon circle" "teal square" "crimson heart" --grid --output test_grid.png
```

## Model Architecture

| Component | Specification | Output Shape |
|-----------|-------------|--------------|
| Text Encoder | Embedding (128-d) → 2-layer LSTM (256-d hidden) | `(batch, 256)` |
| Conditioning | LSTM final hidden state = image latent vector | `(batch, 256)` |
| Image Decoder | FC → 4×4 → TransposedConv → 8×8 → 16×16 → 32×32 → **64×64** | `(batch, 3, 64, 64)` |
| Activation | Tanh (normalized to [-1, 1]) | — |

**Total parameters**: ~3.7M

**Loss function**: Mean Squared Error (MSE)

**Optimizer**: Adam (β₁=0.5, β₂=0.999, lr=0.0002)

## Training Results

Actual training metrics from the best run:

| Metric | Value |
|--------|-------|
| **Best Validation Loss** | **0.0254** |
| **Best Epoch** | **99 / 100** |
| **Final Training Loss** | 0.0276 |
| **Dataset Size** | 2,220 images (1,760 train, 460 val) |
| **Vocabulary Size** | 136 words |
| **Model Parameters** | 3,700,035 |
| **Device** | CUDA (NVIDIA GPU) |
| **Early Stopping Patience** | 20 epochs |

### Training Progression

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1 | 0.5978 | 0.3893 | New best |
| 10 | 0.2117 | 0.1998 | New best |
| 25 | 0.1318 | 0.1217 | New best |
| 34 | 0.1031 | 0.1024 | New best |
| 45 | 0.0693 | 0.0716 | New best |
| 52 | 0.0575 | 0.0539 | New best |
| 58 | 0.0509 | 0.0482 | New best |
| 62 | 0.0493 | 0.0460 | New best |
| 70 | 0.0431 | 0.0421 | New best |
| 75 | 0.0370 | 0.0358 | New best |
| 88 | 0.0308 | 0.0298 | New best |
| 94 | 0.0296 | 0.0274 | New best |
| **99** | **0.0269** | **0.0254** | **✓ Final best** |

### Loss Curve Analysis

- Training and validation losses track closely throughout, indicating **good generalization**
- No significant overfitting observed (val loss does not diverge from train loss)
- Model continues improving through epoch 99, suggesting capacity for longer training
- Early stopping patience of 20 epochs prevents overfitting while allowing convergence

## Why This Approach Is Different

| Aspect | Naive Approach | This Implementation |
|--------|---------------|---------------------|
| Dataset | Random images with duplicates | **2,000+ unique images, zero duplicates** |
| Train/Val Split | Random 80/20 | **Compositional**: same shape, different colors |
| What It Tests | Can the model copy? | Can the model **compose** concepts? |
| Validation Meaning | Meaningless (similar images held out) | Meaningful (unseen combinations) |
| Generalization | None | Tests if color and shape are learned separately |

## Expected Results

### Good Result (Compositional Learning)
- Training loss: 0.03–0.08
- Validation loss: 0.03–0.08 (close to training)
- Generated "maroon circle" looks like a maroon circle
- **Conclusion**: Model learned "maroon" and "circle" as separate concepts

### Bad Result (Memorization)
- Training loss: 0.01–0.03
- Validation loss: 0.20+ (much higher than training)
- Generated "maroon circle" looks blurry or wrong color
- **Conclusion**: Model memorized training pairs, no composition

## Educational Objectives

This project addresses the learning goals from the course material:

1. **Minimal data requirements**: 2,000+ unique samples sufficient for concept learning
2. **Architectural transparency**: Every layer implemented from scratch
3. **Compositional generalization**: The core challenge of text-to-image models
4. **Proper validation**: Testing generalization, not memorization
5. **Reproducible research**: Fixed seeds, deterministic splits, versioned checkpoints

## Limitations & Future Work

- **Resolution**: 64×64 limits detail; scaling requires deeper decoders
- **Color fidelity**: MSE loss averages colors; perceptual loss would improve accuracy
- **Text encoder**: LSTM is basic; Transformer or CLIP embeddings would improve semantics
- **Stochasticity**: Current model is deterministic; adding a VAE latent space would enable diverse outputs from same prompt
- **Dataset size**: 2,000+ combinations tests basic composition; scaling to 10,000+ would improve robustness

## Citation

> Mahdi, M. (2026). *Building a Simple Text-to-Image Model from Scratch: Fundamentals and Practicalities*. Human-Computer Interaction, Zagazig University.

## Dataset Citation

If using the pre-generated dataset from Kaggle:

> Mahmoud, M. (2026). *Colored Geometrics: A Compositional Shape-Color Dataset for Text-to-Image Generation*. Kaggle. https://www.kaggle.com/datasets/mahamedmahmoud/colored-geometrics

## License

MIT License — free for educational and research use.
