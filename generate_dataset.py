"""
1_generate_dataset.py — Compositional geometric shapes dataset
===============================================================

ZERO DUPLICATES. Each (shape, color) pair appears exactly ONCE.
Shapes fill ENTIRE canvas. ONE shape per image.

Train/val split is COMPOSITIONAL: validation contains unseen (shape, color) combos.
This tests if the model learns color and shape as separate concepts.

Usage:
    python 1_generate_dataset.py --output ./shapes_dataset --size 64
"""

import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────

# 20 shapes for ~2000+ unique combinations with colors
SHAPES = [
    'circle',      'square',       'triangle',     'rectangle',
    'ellipse',     'pentagon',     'hexagon',      'heptagon',
    'octagon',     'nonagon',      'decagon',      'star',
    'heart',       'diamond',      'semicircle',   'oval',
    'cross',       'arrow',        'moon',         'ring',
]

# 100 colors for 20 shapes × 100 colors = 2000 unique images
COLORS = {
    # Standard colors
    'red': (255, 0, 0),         'green': (0, 255, 0),       'blue': (0, 0, 255),
    'yellow': (255, 255, 0),    'purple': (128, 0, 128),    'orange': (255, 165, 0),
    'pink': (255, 192, 203),    'cyan': (0, 255, 255),      'brown': (165, 42, 42),
    'white': (255, 255, 255),   'black': (0, 0, 0),         'gray': (128, 128, 128),

    # Extended palette
    'maroon': (128, 0, 0),      'olive': (128, 128, 0),     'teal': (0, 128, 128),
    'navy': (0, 0, 128),        'lime': (50, 205, 50),      'indigo': (75, 0, 130),
    'violet': (238, 130, 238),  'magenta': (255, 0, 255),   'crimson': (220, 20, 60),
    'coral': (255, 127, 80),    'salmon': (250, 128, 114),  'gold': (255, 215, 0),
    'khaki': (240, 230, 140),   'beige': (245, 245, 220),   'ivory': (255, 255, 240),
    'plum': (221, 160, 221),    'orchid': (218, 112, 214),  'tan': (210, 180, 140),
    'peru': (205, 133, 63),     'sienna': (160, 82, 45),    'chocolate': (210, 105, 30),

    # More distinct colors
    'turquoise': (64, 224, 208),'aquamarine': (127, 255, 212),'skyblue': (135, 206, 235),
    'steelblue': (70, 130, 180),'royalblue': (65, 105, 225), 'slateblue': (106, 90, 205),
    'forestgreen': (34, 139, 34),'seagreen': (46, 139, 87),  'springgreen': (0, 255, 127),
    'chartreuse': (127, 255, 0),'lawngreen': (124, 252, 0),  'yellowgreen': (154, 205, 50),
    'firebrick': (178, 34, 34), 'tomato': (255, 99, 71),     'orangered': (255, 69, 0),
    'darkorange': (255, 140, 0),'goldenrod': (218, 165, 32), 'darkgoldenrod': (184, 134, 11),
    'rosybrown': (188, 143, 143),'saddlebrown': (139, 69, 19),'darkgreen': (0, 100, 0),
    'darkcyan': (0, 139, 139),  'darkblue': (0, 0, 139),     'midnightblue': (25, 25, 112),
    'darkslategray': (47, 79, 79),'dimgray': (105, 105, 105), 'slategray': (112, 128, 144),
    'lightslategray': (119, 136, 153),'lightsteelblue': (176, 196, 222),'powderblue': (176, 224, 230),
    'paleturquoise': (175, 238, 238),'lightcyan': (224, 255, 255),'aliceblue': (240, 248, 255),
    'ghostwhite': (248, 248, 255),'lavender': (230, 230, 250), 'thistle': (216, 191, 216),
    'mistyrose': (255, 228, 225),'antiquewhite': (250, 235, 215),'linen': (250, 240, 230),
    'oldlace': (253, 245, 230), 'floralwhite': (255, 250, 240),'cornsilk': (255, 248, 220),
    'lemonchiffon': (255, 250, 205),'lightyellow': (255, 255, 224),'honeydew': (240, 255, 240),
    'mintcream': (245, 255, 250),'azure': (240, 255, 255),    'snow': (255, 250, 250),
    'seashell': (255, 245, 238),'peachpuff': (255, 218, 185),'bisque': (255, 228, 196),
    'moccasin': (255, 228, 181),'navajowhite': (255, 222, 173),'wheat': (245, 222, 179),
    'burlywood': (222, 184, 135),'darkkhaki': (189, 183, 107),'palegreen': (152, 251, 152),
    'lightgreen': (144, 238, 144),'mediumspringgreen': (0, 250, 154),'mediumseagreen': (60, 179, 113),
    'mediumaquamarine': (102, 205, 170),'cadetblue': (95, 158, 160),'cornflowerblue': (100, 149, 237),
    'deepskyblue': (0, 191, 255),'dodgerblue': (30, 144, 255),'lightseagreen': (32, 178, 170),
    'mediumturquoise': (72, 209, 204),'hotpink': (255, 105, 180),'deeppink': (255, 20, 147),
    'palevioletred': (219, 112, 147),'mediumvioletred': (199, 21, 133),'palegoldenrod': (238, 232, 170),
    'mediumorchid': (186, 85, 211),'mediumpurple': (147, 112, 219),'rebeccapurple': (102, 51, 153),
    'blueviolet': (138, 43, 226),'darkorchid': (153, 50, 204),'darkviolet': (148, 0, 211),
}

# ── SHAPE DRAWERS (fill ENTIRE canvas) ────────────────────────────────────

def draw_circle(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, img_size, img_size], fill=color)
    return img

def draw_square(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img_size, img_size], fill=color)
    return img

def draw_rectangle(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img_size, img_size], fill=color)
    return img

def draw_triangle(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 2
    points = [
        (img_size // 2, margin),
        (margin, img_size - margin),
        (img_size - margin, img_size - margin)
    ]
    draw.polygon(points, fill=color)
    return img

def draw_ellipse(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, img_size, img_size], fill=color)
    return img

def draw_pentagon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    r = img_size // 2 - 2
    points = [(cx + r*np.cos(2*np.pi*i/5 - np.pi/2), cy + r*np.sin(2*np.pi*i/5 - np.pi/2)) for i in range(5)]
    draw.polygon(points, fill=color)
    return img

def draw_hexagon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    r = img_size // 2 - 2
    points = [(cx + r*np.cos(2*np.pi*i/6), cy + r*np.sin(2*np.pi*i/6)) for i in range(6)]
    draw.polygon(points, fill=color)
    return img

def draw_heptagon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    r = img_size // 2 - 2
    points = [(cx + r*np.cos(2*np.pi*i/7 - np.pi/2), cy + r*np.sin(2*np.pi*i/7 - np.pi/2)) for i in range(7)]
    draw.polygon(points, fill=color)
    return img

def draw_octagon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    r = img_size // 2 - 2
    points = [(cx + r*np.cos(2*np.pi*i/8 - np.pi/2), cy + r*np.sin(2*np.pi*i/8 - np.pi/2)) for i in range(8)]
    draw.polygon(points, fill=color)
    return img

def draw_nonagon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    r = img_size // 2 - 2
    points = [(cx + r*np.cos(2*np.pi*i/9 - np.pi/2), cy + r*np.sin(2*np.pi*i/9 - np.pi/2)) for i in range(9)]
    draw.polygon(points, fill=color)
    return img

def draw_decagon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    r = img_size // 2 - 2
    points = [(cx + r*np.cos(2*np.pi*i/10 - np.pi/2), cy + r*np.sin(2*np.pi*i/10 - np.pi/2)) for i in range(10)]
    draw.polygon(points, fill=color)
    return img

def draw_star(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    outer_r = img_size // 2 - 2
    inner_r = outer_r // 2
    points = []
    for i in range(10):
        angle = np.pi * i / 5 - np.pi/2
        r = outer_r if i % 2 == 0 else inner_r
        points.append((cx + r*np.cos(angle), cy + r*np.sin(angle)))
    draw.polygon(points, fill=color)
    return img

def draw_heart(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = img_size // 2, img_size // 2
    scale = img_size / 64
    # Heart shape using parametric equations
    points = []
    for i in range(100):
        t = 2 * np.pi * i / 100
        x = 16 * np.sin(t)**3
        y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        px = cx + x * scale * 1.8
        py = cy + y * scale * 1.8 + img_size * 0.1
        points.append((px, py))
    draw.polygon(points, fill=color)
    return img

def draw_diamond(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 2
    points = [
        (img_size // 2, margin),
        (img_size - margin, img_size // 2),
        (img_size // 2, img_size - margin),
        (margin, img_size // 2)
    ]
    draw.polygon(points, fill=color)
    return img

def draw_semicircle(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Upper semicircle
    draw.pieslice([0, 0, img_size, img_size*2], 0, 180, fill=color)
    return img

def draw_oval(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin*2, img_size-margin, img_size-margin*2], fill=color)
    return img

def draw_cross(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    thickness = img_size // 3
    # Horizontal bar
    draw.rectangle([0, (img_size-thickness)//2, img_size, (img_size+thickness)//2], fill=color)
    # Vertical bar
    draw.rectangle([(img_size-thickness)//2, 0, (img_size+thickness)//2, img_size], fill=color)
    return img

def draw_arrow(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Arrow pointing right
    shaft_y = img_size // 2
    shaft_thickness = img_size // 5
    head_size = img_size // 3
    # Shaft
    draw.rectangle([0, shaft_y - shaft_thickness//2, img_size - head_size, shaft_y + shaft_thickness//2], fill=color)
    # Head (triangle)
    points = [
        (img_size - head_size, shaft_y - head_size),
        (img_size, shaft_y),
        (img_size - head_size, shaft_y + head_size)
    ]
    draw.polygon(points, fill=color)
    return img

def draw_moon(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Crescent moon: two overlapping circles
    offset = img_size // 5
    draw.ellipse([0, 0, img_size, img_size], fill=color)
    # Cut out with white circle
    draw.ellipse([offset, offset//2, img_size+offset//2, img_size+offset//2], fill=(255, 255, 255))
    return img

def draw_ring(img_size, color):
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    outer = img_size // 2 - 2
    inner = outer // 2
    # Outer circle
    draw.ellipse([img_size//2-outer, img_size//2-outer, img_size//2+outer, img_size//2+outer], fill=color)
    # Inner circle (white to create ring)
    draw.ellipse([img_size//2-inner, img_size//2-inner, img_size//2+inner, img_size//2+inner], fill=(255, 255, 255))
    return img

SHAPE_DRAWERS = {
    'circle': draw_circle, 'square': draw_square, 'triangle': draw_triangle,
    'rectangle': draw_rectangle, 'ellipse': draw_ellipse, 'pentagon': draw_pentagon,
    'hexagon': draw_hexagon, 'heptagon': draw_heptagon, 'octagon': draw_octagon,
    'nonagon': draw_nonagon, 'decagon': draw_decagon, 'star': draw_star,
    'heart': draw_heart, 'diamond': draw_diamond, 'semicircle': draw_semicircle,
    'oval': draw_oval, 'cross': draw_cross, 'arrow': draw_arrow,
    'moon': draw_moon, 'ring': draw_ring,
}

# ── COMPOSITIONAL SPLIT ───────────────────────────────────────────────────

def compositional_split(combinations, train_ratio=0.8, seed=42):
    """
    Split combinations so that validation has UNSEEN (shape, color) pairs.

    Strategy:
    - For each shape, hold back ~20% of its colors for validation
    - This ensures validation has shapes it has seen, but with NEW colors
    - Tests compositional generalization: can the model combine learned shape + learned color?
    """
    random.seed(seed)

    # Group by shape
    shape_to_colors = {}
    for shape, color in combinations:
        shape_to_colors.setdefault(shape, []).append(color)

    train_combos = []
    val_combos = []

    for shape, colors in shape_to_colors.items():
        random.shuffle(colors)
        n_train = max(1, int(len(colors) * train_ratio))
        train_colors = colors[:n_train]
        val_colors = colors[n_train:]

        for c in train_colors:
            train_combos.append((shape, c))
        for c in val_colors:
            val_combos.append((shape, c))

    random.shuffle(train_combos)
    random.shuffle(val_combos)

    return train_combos, val_combos


# ── GENERATION ─────────────────────────────────────────────────────────────

def generate_dataset(output_dir, img_size, train_ratio=0.8):
    """
    Generate dataset with ZERO duplicates and COMPOSITIONAL train/val split.

    Each (shape, color) pair appears EXACTLY ONCE.
    Validation contains shapes seen in training but with UNSEEN colors.
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train' / 'images'
    val_dir = output_dir / 'val' / 'images'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Generate all unique combinations
    all_combinations = [(shape, color) for shape in SHAPES for color in COLORS.keys()]
    random.shuffle(all_combinations)

    print(f"Total unique combinations: {len(all_combinations)}")
    print(f"Shapes: {len(SHAPES)} ({', '.join(SHAPES)})")
    print(f"Colors: {len(COLORS)} ({', '.join(list(COLORS.keys())[:10])}...)")

    # Compositional split
    train_combos, val_combos = compositional_split(all_combinations, train_ratio)

    print(f"\nTrain set: {len(train_combos)} images (seen shape-color pairs)")
    print(f"Val set:   {len(val_combos)} images (seen shapes, NEW colors)")
    print(f"Split ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")

    # Show examples of held-out colors per shape
    print(f"\nValidation examples (unseen combinations):")
    val_by_shape = {}
    for shape, color in val_combos[:10]:
        print(f"  {shape}: {color} (unseen in training)")

    # Generate train images
    train_captions = {}
    for i, (shape, color) in enumerate(train_combos):
        img, caption = generate_shape_image(img_size, shape, color)
        fname = f"train_{i:05d}.png"
        img.save(train_dir / fname)
        train_captions[fname] = caption

    # Generate val images
    val_captions = {}
    for i, (shape, color) in enumerate(val_combos):
        img, caption = generate_shape_image(img_size, shape, color)
        fname = f"val_{i:05d}.png"
        img.save(val_dir / fname)
        val_captions[fname] = caption

    # Save metadata
    with open(output_dir / 'train_captions.json', 'w', encoding='utf-8') as f:
        json.dump(train_captions, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'val_captions.json', 'w', encoding='utf-8') as f:
        json.dump(val_captions, f, indent=2, ensure_ascii=False)

    # Build vocab from train only (model should not see val vocab during training)
    vocab = build_vocab(train_captions)
    with open(output_dir / 'vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    # Save split info
    split_info = {
        'total_combinations': len(all_combinations),
        'train_size': len(train_combos),
        'val_size': len(val_combos),
        'num_shapes': len(SHAPES),
        'num_colors': len(COLORS),
        'shapes': SHAPES,
        'colors': list(COLORS.keys()),
        'val_examples': [{'shape': s, 'color': c} for s, c in val_combos[:20]],
    }
    with open(output_dir / 'split_info.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Train images: {train_dir} ({len(train_captions)} images)")
    print(f"Val images:   {val_dir} ({len(val_captions)} images)")
    print(f"Vocab size:   {len(vocab)} words")
    print(f"\nKey feature: Validation has {len(val_combos)} UNSEEN (shape, color) pairs")
    print(f"This tests if the model learns color and shape as SEPARATE concepts.")
    print(f"{'='*60}")


def generate_shape_image(img_size, shape_name, color_name):
    """Generate ONE image with ONE shape filling ENTIRE canvas."""
    color_rgb = COLORS[color_name]
    img = SHAPE_DRAWERS[shape_name](img_size, color_rgb)
    caption = f"a {color_name} {shape_name}"
    return img, caption


def build_vocab(captions):
    words = set()
    for cap in captions.values():
        words.update(cap.lower().split())
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    for i, word in enumerate(sorted(words), 4):
        vocab[word] = i
    return vocab


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate compositional shapes dataset')
    parser.add_argument('--output', type=str, default='./shapes_dataset',
                        help='Output directory')
    parser.add_argument('--size', type=int, default=64,
                        help='Image size in pixels (shape fills entire canvas)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of colors per shape for training (0.8 = 80%)')

    args = parser.parse_args()
    generate_dataset(args.output, args.size, args.train_ratio)


if __name__ == '__main__':
    main()