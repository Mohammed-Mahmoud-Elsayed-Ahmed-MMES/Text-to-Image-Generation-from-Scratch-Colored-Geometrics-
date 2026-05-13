"""
3_generate_image.py — Generate images from text prompts
=========================================================

Features:
- Load any checkpoint (best, final, or periodic)
- Test compositional generalization: prompt with unseen combinations
- Compare best vs final model
- Grid generation for multiple prompts

Usage:
    # Basic generation
    python 3_generate_image.py --prompt "a red circle" --model ./checkpoints/best_model.pt

    # Test UNSEEN combination (compositional generalization)
    python 3_generate_image.py --prompt "a maroon circle" --model ./checkpoints/best_model.pt

    # Compare models
    python 3_generate_image.py --compare --prompt "a teal square"

    # Grid of prompts
    python 3_generate_image.py --prompts "red circle" "maroon circle" "teal square" "crimson heart" --grid
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── MODEL ARCHITECTURE ──────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.2)
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        _, (hidden, _) = self.lstm(embedded)
        return hidden[-1]

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, 4, 4)
        return self.deconv(x)

class TextToImageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, text_hidden=256, img_hidden=256):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, embed_dim, text_hidden)
        self.img_decoder = ImageDecoder(text_hidden, img_hidden)
    def forward(self, tokens):
        text_features = self.text_encoder(tokens)
        return self.img_decoder(text_features)


# ── UTILITIES ─────────────────────────────────────────────────────────────

def find_available_model(path):
    path = Path(path)
    if path.exists():
        return path

    checkpoint_dir = path.parent if path.parent != Path('.') else Path('./checkpoints')
    if checkpoint_dir.exists():
        files = sorted(checkpoint_dir.glob('*.pt'))
        if files:
            print(f"\n[!] Model not found: {path}")
            print(f"Available checkpoints:")
            for i, f in enumerate(files, 1):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  {i}. {f.name} ({size_mb:.1f} MB)")
            print(f"\nSuggestion: --model {files[0]}")
            return None

    print(f"\nERROR: No model found. Train first:")
    print(f"  python 2_train_model.py --data_dir ./shapes_dataset")
    return None

def load_model(model_path, device):
    model_path = find_available_model(model_path)
    if model_path is None:
        raise FileNotFoundError("No checkpoint found.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab = checkpoint['vocab']

    model = TextToImageModel(len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"Loaded: {model_path.name}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']+1}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    if 'train_loss' in checkpoint:
        print(f"  Train Loss: {checkpoint['train_loss']:.4f}")

    return model, vocab

def tokenize_prompt(prompt, vocab, max_len=20):
    words = prompt.lower().split()
    tokens = [vocab.get('<sos>', 2)]
    for word in words:
        tokens.append(vocab.get(word, vocab.get('<unk>', 1)))
    tokens.append(vocab.get('<eos>', 3))
    while len(tokens) < max_len:
        tokens.append(vocab.get('<pad>', 0))
    return torch.tensor([tokens], dtype=torch.long)

def tensor_to_pil(tensor):
    img = tensor.cpu().detach()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def generate_single(prompt, model, vocab, device):
    tokens = tokenize_prompt(prompt, vocab).to(device)
    with torch.no_grad():
        generated = model(tokens)
    return tensor_to_pil(generated[0])

def create_grid(prompts, model, vocab, device, cols=5):
    images = []
    for prompt in prompts:
        img = generate_single(prompt, model, vocab, device)
        images.append((prompt, img))

    n = len(images)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    w, h = images[0][1].size
    text_h = 25
    grid_w = w * cols
    grid_h = (h + text_h) * rows

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for i, (prompt, img) in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * w
        y = row * (h + text_h)
        grid.paste(img, (x, y + text_h))
        draw.text((x + 5, y + 5), prompt[:25], fill=(0, 0, 0), font=font)

    return grid

def compare_models(prompt, best_path, final_path, device, output='comparison.png'):
    print("\n=== COMPARING BEST vs FINAL ===")

    best_model, best_vocab = load_model(best_path, device)
    best_img = generate_single(prompt, best_model, best_vocab, device)

    final_model, final_vocab = load_model(final_path, device)
    final_img = generate_single(prompt, final_model, final_vocab, device)

    w, h = best_img.size
    comparison = Image.new('RGB', (w * 2 + 20, h + 40), (255, 255, 255))
    draw = ImageDraw.Draw(comparison)

    comparison.paste(best_img, (5, 30))
    comparison.paste(final_img, (w + 15, 30))

    draw.text((5, 5), f"BEST: {prompt}", fill=(0, 128, 0))
    draw.text((w + 15, 5), f"FINAL: {prompt}", fill=(128, 0, 0))

    comparison.save(output)
    print(f"Saved: {output}")


def test_compositional(model, vocab, device, data_dir, num_tests=10):
    """
    Test compositional generalization by generating unseen combinations.
    Loads val_captions.json to find prompts the model never saw during training.
    """
    print("\n=== COMPOSITIONAL GENERALIZATION TEST ===")
    print("Testing if model can generate unseen (shape, color) combinations...")

    data_dir = Path(data_dir)
    val_file = data_dir / 'val_captions.json'

    if not val_file.exists():
        print("No val_captions.json found. Generate dataset first.")
        return

    with open(val_file, 'r') as f:
        val_captions = json.load(f)

    # Sample random validation prompts
    val_prompts = list(val_captions.values())
    test_prompts = random.sample(val_prompts, min(num_tests, len(val_prompts)))

    print(f"\nGenerating {len(test_prompts)} UNSEEN combinations:")
    for p in test_prompts:
        print(f"  - {p}")

    grid = create_grid(test_prompts, model, vocab, device, cols=5)
    grid.save('compositional_test.png')
    print(f"\nSaved: compositional_test.png")
    print("If these look correct, the model learned color and shape as separate concepts!")


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate images from text')

    parser.add_argument('--prompt', type=str, help='Single prompt')
    parser.add_argument('--prompts', type=str, nargs='+', help='Multiple prompts')
    parser.add_argument('--file', type=str, help='File with prompts')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pt',
                        help='Model checkpoint path')
    parser.add_argument('--compare', action='store_true', help='Compare best vs final')
    parser.add_argument('--test_compositional', action='store_true',
                        help='Test on unseen validation combinations')
    parser.add_argument('--data_dir', type=str, default='./shapes_dataset',
                        help='Dataset dir (for compositional test)')
    parser.add_argument('--output', type=str, default='generated.png')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--device', type=str, help='cuda, mps, or cpu')

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Compare mode
    if args.compare:
        ckpt_path = Path(args.model)
        best_path = ckpt_path.parent / 'best_model.pt'
        final_path = ckpt_path.parent / 'final_model.pt'
        prompt = args.prompt or "a red circle"
        try:
            compare_models(prompt, best_path, final_path, device, args.output)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return

    # Load model
    try:
        print(f"\nLoading: {args.model}")
        model, vocab = load_model(args.model, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Compositional test mode
    if args.test_compositional:
        test_compositional(model, vocab, device, args.data_dir)
        return

    # Collect prompts
    prompts = []
    if args.prompt: prompts.append(args.prompt)
    if args.prompts: prompts.extend(args.prompts)
    if args.file:
        with open(args.file, 'r') as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if not prompts:
        prompts = ["a red circle"]

    print(f"\nGenerating {len(prompts)} image(s)...")

    if len(prompts) == 1 and not args.grid:
        img = generate_single(prompts[0], model, vocab, device)
        img.save(args.output)
        print(f"Saved: {args.output}")
    else:
        grid = create_grid(prompts, model, vocab, device)
        grid.save(args.output)
        print(f"Grid saved: {args.output}")

    print("Done!")

if __name__ == '__main__':
    main()