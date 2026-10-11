"""
Evaluate shape vs texture bias of the trained VGG model on cue-conflict images.

Loads conflict images from conflicts/<shape>_<texture>/NNN.png,
runs the model, and records whether each prediction matches
the shape class (content) or texture class (style).

Outputs:
  results/results.csv        — per-image predictions
  results/bias_scores.csv    — per-pair shape bias scores
  results/heatmap.png        — 10×10 shape bias heatmap
  results/summary.txt        — overall statistics
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

from train import VGGNet, CLASSES


def load_model(checkpoint_path, device):
    model = VGGNet().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt["epoch"]}, test acc {ckpt["test_acc"]:.4f})')
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066),
                             (0.2603, 0.2566, 0.2713)),
    ])


@torch.no_grad()
def predict_image(model, img_path, transform, device):
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    return logits.argmax(1).item()


def collect_predictions(model, conflicts_dir, transform, device):
    records = []

    for pair_name in sorted(os.listdir(conflicts_dir)):
        pair_dir = os.path.join(conflicts_dir, pair_name)
        if not os.path.isdir(pair_dir):
            continue

        parts = pair_name.split('_', 1)
        if len(parts) != 2:
            continue
        shape_name, tex_name = parts[0], parts[1]

        if shape_name not in CLASSES or tex_name not in CLASSES:
            continue

        shape_idx = CLASSES.index(shape_name)
        tex_idx   = CLASSES.index(tex_name)

        images = sorted(f for f in os.listdir(pair_dir) if f.endswith('.png'))
        for fname in images:
            pred = predict_image(model,
                                 os.path.join(pair_dir, fname),
                                 transform, device)
            records.append({
                'shape_class':   shape_name,
                'texture_class': tex_name,
                'shape_idx':     shape_idx,
                'texture_idx':   tex_idx,
                'predicted_idx': pred,
                'predicted':     CLASSES[pred],
                'is_shape':      int(pred == shape_idx),
                'is_texture':    int(pred == tex_idx),
            })

    return pd.DataFrame(records)


def compute_bias_scores(df):
    """
    Shape bias = shape_correct / (shape_correct + texture_correct)
    Rows where prediction is neither are excluded from the ratio
    but tracked separately.
    """
    rows = []
    for (shape, texture), group in df.groupby(['shape_class', 'texture_class']):
        n_total   = len(group)
        n_shape   = group['is_shape'].sum()
        n_texture = group['is_texture'].sum()
        n_neither = n_total - n_shape - n_texture
        denom     = n_shape + n_texture
        bias      = n_shape / denom if denom > 0 else float('nan')
        rows.append({
            'shape_class':   shape,
            'texture_class': texture,
            'n_total':       n_total,
            'n_shape':       n_shape,
            'n_texture':     n_texture,
            'n_neither':     n_neither,
            'shape_bias':    bias,
        })
    return pd.DataFrame(rows)


def plot_heatmap(bias_df, out_path):
    matrix = np.full((len(CLASSES), len(CLASSES)), np.nan)

    for _, row in bias_df.iterrows():
        r = CLASSES.index(row['shape_class'])
        c = CLASSES.index(row['texture_class'])
        matrix[r, c] = row['shape_bias']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Shape bias score (1 = shape, 0 = texture)')

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Texture class (style donor)')
    ax.set_ylabel('Shape class (content donor)')
    ax.set_title('Shape Bias Score per Cue-Conflict Pair\n'
                 '(green = shape-biased, red = texture-biased)')

    # Annotate cells
    for r in range(len(CLASSES)):
        for c in range(len(CLASSES)):
            if not np.isnan(matrix[r, c]):
                ax.text(c, r, f'{matrix[r, c]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Heatmap saved to {out_path}')


def write_summary(df, bias_df, out_path):
    n_shape   = df['is_shape'].sum()
    n_texture = df['is_texture'].sum()
    n_neither = len(df) - n_shape - n_texture
    denom     = n_shape + n_texture
    overall   = n_shape / denom if denom > 0 else float('nan')

    lines = [
        'Shape vs Texture Bias — Summary',
        '=' * 40,
        f'Total conflict images evaluated : {len(df)}',
        f'Predicted shape class           : {n_shape} ({n_shape/len(df)*100:.1f}%)',
        f'Predicted texture class         : {n_texture} ({n_texture/len(df)*100:.1f}%)',
        f'Predicted neither               : {n_neither} ({n_neither/len(df)*100:.1f}%)',
        f'',
        f'Overall shape bias score        : {overall:.4f}',
        f'  (1.0 = fully shape-biased, 0.0 = fully texture-biased, 0.5 = no preference)',
        f'',
        'Top 5 most shape-biased pairs:',
    ]
    top_shape = bias_df.nlargest(5, 'shape_bias')[
        ['shape_class', 'texture_class', 'shape_bias']]
    for _, r in top_shape.iterrows():
        lines.append(f'  {r.shape_class:10s} + {r.texture_class:10s} texture -> {r.shape_bias:.3f}')

    lines += ['', 'Top 5 most texture-biased pairs:']
    top_tex = bias_df.nsmallest(5, 'shape_bias')[
        ['shape_class', 'texture_class', 'shape_bias']]
    for _, r in top_tex.iterrows():
        lines.append(f'  {r.shape_class:10s} + {r.texture_class:10s} texture -> {r.shape_bias:.3f}')

    text = '\n'.join(lines)
    print('\n' + text)
    with open(out_path, 'w') as f:
        f.write(text + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   default='checkpoints/best.pth')
    parser.add_argument('--conflicts-dir',default='conflicts')
    parser.add_argument('--out-dir',      default='results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)

    model     = load_model(args.checkpoint, device)
    transform = get_transform()

    print('Running predictions on conflict images...')
    df = collect_predictions(model, args.conflicts_dir, transform, device)

    if df.empty:
        print(f'No conflict images found in {args.conflicts_dir}. '
              'Run generate_conflicts.py first.')
        return

    df.to_csv(os.path.join(args.out_dir, 'results.csv'), index=False)
    print(f'Saved {len(df)} predictions to results/results.csv')

    bias_df = compute_bias_scores(df)
    bias_df.to_csv(os.path.join(args.out_dir, 'bias_scores.csv'), index=False)

    plot_heatmap(bias_df, os.path.join(args.out_dir, 'heatmap.png'))
    write_summary(df, bias_df, os.path.join(args.out_dir, 'summary.txt'))


if __name__ == '__main__':
    main()
