# Text-to-Image Generation from Scratch

A minimal, end-to-end implementation of a text-to-image generative model built entirely from scratch using PyTorch. This project demonstrates **compositional generalization** — the ability to generate unseen combinations of learned concepts (e.g., generating a "maroon circle" after training on "red circle" and "maroon square").

## Key Feature: Compositional Generalization

Unlike naive approaches that memorize exact (image, caption) pairs, this implementation:
- Uses **zero duplicate images** — each (shape, color) combination appears exactly once
- Splits data **compositionally**: training sees some colors for each shape, validation tests **unseen colors** for the same shapes
- Tests whether the model learns "red", "circle", "maroon" as **separate composable concepts**

## Repository Structure

```
├── 1_generate_dataset.py      # Generate 2,000 unique shape-color combinations
├── 2_train_model.py           # Train with compositional validation split
├── 3_generate_image.py        # Generate + test compositional generalization
├── generate.bat               # Windows launcher
├── generate.ps1               # PowerShell launcher
├── README.md                  # This file
├── shapes_dataset/            # Generated dataset
│   ├── train/
│   │   └── images/            # 1,600 images (seen combinations)
│   ├── val/
│   │   └── images/            # 400 images (unseen combinations)
│   ├── train_captions.json
│   ├── val_captions.json
│   ├── vocab.json
│   └── split_info.json        # Documents which colors are held out per shape
└── checkpoints/
    ├── best_model.pt          # Best validation performance
    ├── final_model.pt
    ├── checkpoint_epoch{10,20,...}.pt
    ├── loss_curve.png
    └── visualizations/
        └── epoch_001.png      # Shows seen vs unseen prompt generation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- Pillow, NumPy, Matplotlib

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Quick Start

### Step 1: Generate Compositional Dataset

Creates **2,000 unique images** (20 shapes × 100 colors) with **zero duplicates**.

```bash
python 1_generate_dataset.py --output ./shapes_dataset --size 64
```

**Dataset composition:**
- **20 shapes**: circle, square, triangle, rectangle, ellipse, pentagon, hexagon, heptagon, octagon, nonagon, decagon, star, heart, diamond, semicircle, oval, cross, arrow, moon, ring
- **100 colors**: red, blue, green, yellow, purple, orange, pink, cyan, brown, white, black, gray, maroon, olive, teal, navy, lime, indigo, violet, magenta, crimson, coral, salmon, gold, khaki, beige, ivory, plum, orchid, tan, peru, sienna, chocolate, turquoise, aquamarine, skyblue, steelblue, royalblue, slateblue, forestgreen, seagreen, springgreen, chartreuse, lawngreen, yellowgreen, firebrick, tomato, orangered, darkorange, goldenrod, darkgoldenrod, rosybrown, saddlebrown, darkgreen, darkcyan, darkblue, midnightblue, darkslategray, dimgray, slategray, lightslategray, lightsteelblue, powderblue, paleturquoise, lightcyan, aliceblue, ghostwhite, lavender, thistle, mistyrose, antiquewhite, linen, oldlace, floralwhite, cornsilk, lemonchiffon, lightyellow, honeydew, mintcream, azure, snow, seashell, peachpuff, bisque, moccasin, navajowhite, wheat, burlywood, darkkhaki, palegreen, lightgreen, mediumspringgreen, mediumseagreen, mediumaquamarine, cadetblue, cornflowerblue, deepskyblue, dodgerblue, lightseagreen, mediumturquoise, hotpink, deeppink, palevioletred, mediumvioletred, palegoldenrod, mediumorchid, mediumpurple, rebeccapurple, blueviolet, darkorchid, darkviolet

**Compositional split:**
- **Train (80%)**: 1,600 images — for each shape, 80 colors are shown
- **Val (20%)**: 400 images — the remaining 20 colors per shape are **held out**

Example held-out combinations (unseen in training):
```json
{"shape": "circle", "color": "maroon"}   // circle seen, but not maroon
{"shape": "square", "color": "teal"}     // square seen, but not teal
{"shape": "heart", "color": "crimson"}   // heart seen, but not crimson
```

### Step 2: Train the Model

```bash
python 2_train_model.py --data_dir ./shapes_dataset --epochs 100 --patience 20
```

**What happens during training:**
- Model learns from **seen** (shape, color) pairs
- Validation tests on **unseen** (shape, color) pairs
- If validation loss improves, model is learning compositional concepts
- If validation loss diverges, model is just memorizing

**Console output:**
```
Epoch 15 | Train Loss: 0.0456 | Val Loss: 0.0891
  *** NEW BEST *** Val Loss: 0.0891 (epoch 15)

Epoch 35 | Train Loss: 0.0321 | Val Loss: 0.1567
  No improvement: 20/20 (best: 0.0891 at 15)

*** EARLY STOPPING ***
Best model: epoch 15, val_loss=0.0891
```

### Step 3: Generate Images

**Basic generation (seen combination):**
```bash
python 3_generate_image.py --prompt "a red circle" --model ./checkpoints/best_model.pt
```

**Test compositional generalization (unseen combination):**
```bash
# Generate a specific unseen combo
python 3_generate_image.py --prompt "a maroon circle" --model ./checkpoints/best_model.pt

# Batch test on all held-out validation combinations
python 3_generate_image.py --test_compositional --data_dir ./shapes_dataset --model ./checkpoints/best_model.pt
```

**Compare best vs final model:**
```bash
python 3_generate_image.py --compare --prompt "a teal square"
```

**Grid generation:**
```bash
python 3_generate_image.py --prompts "red circle" "maroon circle" "teal square" "crimson heart" --grid --output test_grid.png
```

## Model Architecture

| Component | Specification | Output |
|-----------|---------------|--------|
| Text Encoder | Embedding (128-d) → 2-layer LSTM (256-d) | `(batch, 256)` |
| Conditioning | LSTM hidden state → image latent vector | `(batch, 256)` |
| Image Decoder | FC → 4×4 → TransposedConv → 8×8 → 16×16 → 32×32 → **64×64** | `(batch, 3, 64, 64)` |
| Activation | Tanh | `[-1, 1]` |

**Parameters**: ~2.5M

**Loss**: MSE (reconstruction)

**Optimizer**: Adam (lr=0.0002, β₁=0.5, β₂=0.999)

## Why This Approach Is Different

| Aspect | Naive Approach | This Implementation |
|--------|---------------|---------------------|
| Dataset | 5,000 random images, many duplicates | 2,000 unique images, **zero duplicates** |
| Train/Val Split | Random 80/20 | **Compositional**: same shape, different colors |
| What It Tests | Can the model copy? | Can the model **compose** concepts? |
| Validation Meaning | Meaningless (same images held out) | Meaningful (unseen combinations) |
| Generalization | None | Tests if color and shape are learned separately |

## Expected Results

### Good Result (Compositional Learning)
- Training loss: 0.03–0.08
- Validation loss: 0.06–0.12 (close to training)
- Generated "maroon circle" looks like a maroon circle
- **Conclusion**: Model learned "maroon" and "circle" as separate concepts

### Bad Result (Memorization)
- Training loss: 0.01–0.03
- Validation loss: 0.20+ (much higher than training)
- Generated "maroon circle" looks blurry or wrong color
- **Conclusion**: Model memorized training pairs, no composition

## Educational Objectives

This project addresses the learning goals from the course material:

1. **Minimal data requirements**: 2,000 unique samples sufficient for concept learning
2. **Architectural transparency**: Every layer implemented from scratch
3. **Compositional generalization**: The core challenge of text-to-image models
4. **Proper validation**: Testing generalization, not memorization
5. **Reproducible research**: Fixed seeds, deterministic splits, versioned checkpoints

## Limitations & Future Work

- **Resolution**: 64×64 limits detail; scaling requires deeper decoders
- **Color fidelity**: MSE loss averages colors; perceptual loss would improve accuracy
- **Text encoder**: LSTM is basic; Transformer or CLIP embeddings would improve semantics
- **Stochasticity**: Current model is deterministic; adding a VAE latent space would enable diverse outputs from same prompt
- **Dataset size**: 2,000 combinations tests basic composition; scaling to 10,000+ would improve robustness

## Citation

> Mahdi, M. (2026). *Building a Simple Text-to-Image Model from Scratch: Fundamentals and Practicalities*. Human-Computer Interaction, Zagazig University.

## License

MIT License — free for educational and research use.
