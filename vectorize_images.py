import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import normalize
import torch
import torchvision.transforms as T

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VECTORS_DIR = os.path.join(BASE_DIR, 'data', 'vectors')

DATASETS = {
    'primary': {
        'images_dir': os.path.join(BASE_DIR, 'data', 'dataset_A_images_primary'),
        'npz_out':    os.path.join(VECTORS_DIR, 'image_vectors.npz'),
        'json_out':   os.path.join(VECTORS_DIR, 'image_vectors.json'),
    },
    'secondary': {
        'images_dir': os.path.join(BASE_DIR, 'data', 'dataset_A_images_secondary'),
        'npz_out':    os.path.join(VECTORS_DIR, 'image_vectors_secondary.npz'),
        'json_out':   None,
    },
}

DINOV2_MODEL = 'dinov2_vitb14'   # options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
BATCH_SIZE   = 16
DEVICE       = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# DINOv2 expects 224x224 images normalized with ImageNet stats
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> torch.Tensor | None:
    try:
        img = Image.open(path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


# ============================================================
# VECTORIZE
# ============================================================

def vectorize(model, images_dir: str) -> dict[str, list]:
    image_paths = (sorted(Path(images_dir).glob('*.jpeg')) +
                   sorted(Path(images_dir).glob('*.jpg')) +
                   sorted(Path(images_dir).glob('*.png')))
    print(f"  Found {len(image_paths)} images in {images_dir}")

    all_vectors: dict[str, list] = {}
    batch_names: list[str] = []
    batch_tensors: list[torch.Tensor] = []

    def flush_batch():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        for name, vec in zip(batch_names, embeddings):
            all_vectors[name] = vec.tolist()
        batch_names.clear()
        batch_tensors.clear()

    for i, path in enumerate(image_paths):
        tensor = load_image(str(path))
        if tensor is None:
            continue
        batch_names.append(path.name)
        batch_tensors.append(tensor)
        if len(batch_tensors) >= BATCH_SIZE:
            flush_batch()
            print(f"  Processed {i + 1}/{len(image_paths)}")

    flush_batch()
    print(f"  Processed {len(all_vectors)}/{len(image_paths)} images")
    return all_vectors


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Vectorize auction images with DINOv2')
    parser.add_argument('--dataset', choices=['primary', 'secondary', 'both'], default='primary',
                        help='Which dataset to vectorize (default: primary)')
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == 'both' else [args.dataset]

    os.makedirs(VECTORS_DIR, exist_ok=True)
    print(f"Loading {DINOV2_MODEL} on {DEVICE}...")
    model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL)
    model.eval().to(DEVICE)

    for dataset_name in datasets:
        cfg = DATASETS[dataset_name]
        print(f"\n[{dataset_name}] Vectorizing...")
        vectors = vectorize(model, cfg['images_dir'])

        if not vectors:
            print(f"  No vectors produced for {dataset_name}, skipping save.")
            continue

        # L2-normalise
        names  = list(vectors.keys())
        matrix = normalize(np.stack([np.array(vectors[n]) for n in names]), norm='l2').astype(np.float32)

        # Save NPZ
        np.savez_compressed(cfg['npz_out'], **dict(zip(names, matrix)))
        print(f"  Saved NPZ → {cfg['npz_out']}")

        # Save JSON (primary only)
        if cfg['json_out']:
            with open(cfg['json_out'], 'w') as f:
                json.dump({n: matrix[i].tolist() for i, n in enumerate(names)}, f)
            print(f"  Saved JSON → {cfg['json_out']}")

        dim = matrix.shape[1]
        print(f"  Embedding dimension: {dim}  |  Total: {len(vectors)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
