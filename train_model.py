"""
2_train_model.py — Train text-to-image model on compositional dataset
=====================================================================

Trains on seen (shape, color) pairs.
Validates on UNSEEN (shape, color) pairs — same shape, new color.
This tests compositional generalization.

Usage:
    python 2_train_model.py --data_dir ./shapes_dataset --epochs 100 --patience 20
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# DATASET (loads from train/ and val/ folders)
# ═════════════════════════════════════════════════════════════════════════════

class CompositionalDataset(Dataset):
    """Dataset that loads pre-split train/val compositional data."""

    def __init__(self, data_dir, split='train', img_size=64, max_len=20):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_len = max_len

        # Load captions for this split
        cap_file = self.data_dir / f'{split}_captions.json'
        with open(cap_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

        # Load vocab (built from train only)
        with open(self.data_dir / 'vocab.json', 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.image_files = sorted(self.captions.keys())
        self.img_dir = self.data_dir / split / 'images'

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_files)

    def tokenize(self, caption):
        words = caption.lower().split()
        tokens = [self.vocab.get('<sos>', 2)]
        for word in words:
            tokens.append(self.vocab.get(word, self.vocab['<unk>']))
        tokens.append(self.vocab.get('<eos>', 3))
        if len(tokens) < self.max_len:
            tokens += [self.vocab['<pad>']] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len-1] + [self.vocab['<eos>']]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img = Image.open(self.img_dir / fname).convert('RGB')
        img = self.transform(img)
        caption = self.captions[fname]
        tokens = self.tokenize(caption)
        return img, tokens, caption


# ═════════════════════════════════════════════════════════════════════════════
# MODEL (same architecture)
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def tensor_to_image(tensor):
    img = tensor.cpu().detach()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def create_comparison_grid(model, test_prompts, vocab, device, epoch, save_path):
    model.eval()
    images = []
    for prompt in test_prompts:
        words = prompt.lower().split()
        tokens = [vocab.get('<sos>', 2)]
        for word in words:
            tokens.append(vocab.get(word, vocab['<unk>']))
        tokens.append(vocab.get('<eos>', 3))
        while len(tokens) < 20:
            tokens.append(vocab['<pad>'])
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            generated = model(tokens)
        img = tensor_to_image(generated[0])
        images.append((prompt, img))

    n = len(images)
    w, h = images[0][1].size
    grid_w = w * n
    grid_h = h + 30
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for i, (prompt, img) in enumerate(images):
        grid.paste(img, (i * w, 30))
        draw.text((i * w + 5, 5), prompt, fill=(0, 0, 0), font=font)

    draw.text((5, grid_h - 20), f"Epoch {epoch}", fill=(0, 0, 0), font=font)
    grid.save(save_path)
    print(f"  Saved viz: {save_path}")
    model.train()
    return grid

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (seen combos)', linewidth=2)
    plt.plot(val_losses, label='Val Loss (unseen combos)', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress — Compositional Generalization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved loss plot: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train(data_dir, epochs=100, batch_size=64, lr=0.0002,
          device='cuda', save_dir='./checkpoints', patience=20,
          test_prompts=None):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    viz_dir = save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    train_dataset = CompositionalDataset(data_dir, split='train')
    val_dataset = CompositionalDataset(data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0,
                             pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)

    vocab = train_dataset.vocab
    vocab_size = len(vocab)

    print(f"Train: {len(train_dataset)} images (seen shape-color pairs)")
    print(f"Val:   {len(val_dataset)} images (unseen shape-color pairs)")
    print(f"Vocab: {vocab_size} words")

    # Model
    model = TextToImageModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Early stopping patience: {patience}")
    print(f"Max epochs: {epochs}\n")

    # Test prompts: include some validation (unseen) combinations
    if test_prompts is None:
        # Mix of train-seen and val-unseen combos
        test_prompts = [
            "a red circle",      # likely seen in train
            "a blue square",     # likely seen in train
            "a maroon circle",   # likely UNSEEN (val color)
            "a teal square",     # likely UNSEEN (val color)
            "a crimson heart",   # likely UNSEEN (val combo)
        ]

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, (images, tokens, _) in enumerate(train_loader):
            images = images.to(device)
            tokens = tokens.to(device)

            generated = model(tokens)
            loss = criterion(generated, images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # ── VALIDATION (on UNSEEN combinations) ──
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, tokens, _ in val_loader:
                images = images.to(device)
                tokens = tokens.to(device)
                generated = model(tokens)
                loss = criterion(generated, images)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save periodic
        if (epoch + 1) % 10 == 0:
            periodic_path = save_dir / f'checkpoint_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'vocab': vocab,
            }, periodic_path)
            print(f"  Saved checkpoint: {periodic_path}")

        # Save BEST
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0

            best_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'vocab': vocab,
            }, best_path)
            print(f"  *** NEW BEST *** Val Loss: {avg_val_loss:.4f} (epoch {best_epoch})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement: {epochs_no_improve}/{patience} (best: {best_val_loss:.4f} at {best_epoch})")

        # Visualize
        viz_path = viz_dir / f'epoch_{epoch+1:03d}.png'
        create_comparison_grid(model, test_prompts, vocab, device, epoch+1, viz_path)
        plot_losses(train_losses, val_losses, save_dir / 'loss_curve.png')

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n*** EARLY STOPPING ***")
            print(f"No improvement for {patience} epochs.")
            print(f"Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
            break

    # Save final
    final_path = save_dir / 'final_model.pt'
    torch.save({'model_state': model.state_dict(), 'vocab': vocab}, final_path)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"Best model: {save_dir / 'best_model.pt'}")
    print(f"Final model: {final_path}")
    print(f"Visualizations: {viz_dir}")
    print(f"Loss curve: {save_dir / 'loss_curve.png'}")
    print(f"{'='*60}")

    return model, vocab


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train compositional text-to-image')
    parser.add_argument('--data_dir', type=str, default='./shapes_dataset',
                        help='Dataset directory with train/ and val/')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        patience=args.patience
    )


if __name__ == '__main__':
    main()