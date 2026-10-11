"""
Generate cue-conflict images for the shape-vs-texture experiment.

For every ordered class pair (A, B) where A != B:
  - Content image (shape donor): class A test image
  - Style image   (texture donor): class B test image
  - Output: style-transferred image saved to conflicts/<A>_<B>/NNN.png

90 pairs × 50 images = 4,500 conflict images total.

Uses Gatys et al. neural style transfer with a frozen VGG19 feature extractor.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import argparse
from collections import defaultdict

CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
           'dog', 'horse', 'monkey', 'ship', 'truck']

# VGG19 layer names used for content and style losses
CONTENT_LAYERS = ['relu4_2']
STYLE_LAYERS   = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']


# --------------------------------------------------------------------------- #
# VGG19 feature extractor                                                     #
# --------------------------------------------------------------------------- #

class VGG19Features(nn.Module):
    """Return activations at named layers. Input must be ImageNet-normalised."""

    LAYER_MAP = {
        'relu1_1': 1,  'relu1_2': 3,
        'relu2_1': 6,  'relu2_2': 8,
        'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
        'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
        'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
    }

    def __init__(self, layers):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleDict()
        prev = 0
        # must process in ascending index order so slices chain correctly
        for name in sorted(layers, key=lambda n: self.LAYER_MAP[n]):
            idx = self.LAYER_MAP[name]
            self.slices[name] = nn.Sequential(*list(vgg.children())[prev:idx + 1])
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feats, out = {}, x
        for name, slice_ in self.slices.items():
            out = slice_(out)
            feats[name] = out
        return feats


def gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


# --------------------------------------------------------------------------- #
# Style transfer optimisation                                                  #
# --------------------------------------------------------------------------- #

def style_transfer(content_img, style_img, extractor, device,
                   steps=300, content_weight=1.0, style_weight=1e6):
    """
    Returns a PIL image: content_img shape + style_img texture.
    Both inputs are PIL Images; output is a PIL Image at the same size.
    """
    size = content_img.size  # (W, H)

    # ImageNet normalisation for VGG19 feature extractor
    to_tensor = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def load(img):
        return to_tensor(img).unsqueeze(0).to(device)

    content_t = load(content_img)
    style_t   = load(style_img)

    # Target features
    all_layers = CONTENT_LAYERS + STYLE_LAYERS
    ext = VGG19Features(all_layers).to(device)

    with torch.no_grad():
        content_feats = ext(content_t)
        style_feats   = ext(style_t)
        style_grams   = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}

    # Initialise output from content image
    output = content_t.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([output], max_iter=steps, tolerance_grad=1e-7)

    def closure():
        optimizer.zero_grad()
        feats = ext(output)

        c_loss = sum(
            nn.functional.mse_loss(feats[l], content_feats[l])
            for l in CONTENT_LAYERS
        )
        s_loss = sum(
            nn.functional.mse_loss(gram_matrix(feats[l]), style_grams[l])
            for l in STYLE_LAYERS
        )
        loss = content_weight * c_loss + style_weight * s_loss
        loss.backward()
        return loss

    optimizer.step(closure)

    # Denormalise and convert to PIL
    img = output.detach().squeeze(0).cpu()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Data helpers                                                                 #
# --------------------------------------------------------------------------- #

def load_stl10_by_class(data_dir):
    """Return dict: class_idx -> list of PIL Images from the test split."""
    dataset = datasets.STL10(data_dir, split='test', download=True,
                             transform=transforms.ToTensor())
    by_class = defaultdict(list)
    for img_t, label in DataLoader(dataset, batch_size=1):
        pil = transforms.ToPILImage()(img_t.squeeze(0))
        by_class[label.item()].append(pil)
    return by_class


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      default='data')
    parser.add_argument('--out-dir',       default='conflicts')
    parser.add_argument('--n-per-pair',    type=int,   default=50)
    parser.add_argument('--steps',         type=int,   default=300)
    parser.add_argument('--content-weight',type=float, default=1.0)
    parser.add_argument('--style-weight',  type=float, default=1e6)
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading STL-10 test images...')
    by_class = load_stl10_by_class(args.data_dir)

    all_layers = CONTENT_LAYERS + STYLE_LAYERS
    extractor  = VGG19Features(all_layers).to(device)

    total_pairs = len(CLASSES) * (len(CLASSES) - 1)
    done = 0

    for shape_idx, shape_name in enumerate(CLASSES):
        for tex_idx, tex_name in enumerate(CLASSES):
            if shape_idx == tex_idx:
                continue

            pair_dir = os.path.join(args.out_dir, f'{shape_name}_{tex_name}')
            os.makedirs(pair_dir, exist_ok=True)

            # Skip already-completed pairs (resume-friendly)
            existing = len([f for f in os.listdir(pair_dir) if f.endswith('.png')])
            if existing >= args.n_per_pair:
                done += 1
                continue

            content_imgs = by_class[shape_idx][:args.n_per_pair]
            style_imgs   = by_class[tex_idx][:args.n_per_pair]

            for i in range(existing, args.n_per_pair):
                out_path = os.path.join(pair_dir, f'{i:03d}.png')
                result = style_transfer(
                    content_imgs[i], style_imgs[i],
                    extractor, device,
                    steps=args.steps,
                    content_weight=args.content_weight,
                    style_weight=args.style_weight,
                )
                result.save(out_path)

            done += 1
            print(f'[{done}/{total_pairs}] {shape_name} -> {tex_name}')

    print(f'\nDone. Conflict images saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
