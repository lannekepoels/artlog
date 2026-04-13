"""
Full pipeline: vectorize images → build FAISS indices → match secondary against primary.

Usage:
  python faiss_workflow.py                  # vectorize both datasets + match
  python faiss_workflow.py --skip-vectorize # skip vectorization (use existing NPZ files)

Outputs (data/vectors/):
  image_vectors.npz / image_vectors.json   (primary)
  image_vectors_secondary.npz              (secondary)
  primary.faiss / primary_names.npy
  secondary.faiss / secondary_names.npy
  matches_faiss.csv
  matches_faiss.json
"""

import os
import gc
import csv
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as T
import faiss
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import normalize

# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.80

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
VECTORS_DIR     = os.path.join(BASE_DIR, 'data', 'vectors')

DATASETS = {
    'primary': {
        'images_dir': os.path.join(BASE_DIR, 'data', 'dataset_A_images_primary'),
        'npz_out':    os.path.join(VECTORS_DIR, 'image_vectors.npz'),
        'json_out':   os.path.join(VECTORS_DIR, 'image_vectors.json'),
        'index_out':  os.path.join(VECTORS_DIR, 'primary.faiss'),
        'names_out':  os.path.join(VECTORS_DIR, 'primary_names.npy'),
    },
    'secondary': {
        'images_dir': os.path.join(BASE_DIR, 'data', 'dataset_A_images_secondary'),
        'npz_out':    os.path.join(VECTORS_DIR, 'image_vectors_secondary.npz'),
        'json_out':   None,
        'index_out':  os.path.join(VECTORS_DIR, 'secondary.faiss'),
        'names_out':  os.path.join(VECTORS_DIR, 'secondary_names.npy'),
    },
}

DINOV2_MODEL = 'dinov2_vitb14'
BATCH_SIZE   = 16
DEVICE       = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> torch.Tensor | None:
    try:
        return transform(Image.open(path).convert('RGB'))
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


# ============================================================
# STEP 1: VECTORIZE
# ============================================================

def vectorize(model, images_dir: str) -> tuple[list[str], np.ndarray]:
    image_paths = (sorted(Path(images_dir).glob('*.jpeg')) +
                   sorted(Path(images_dir).glob('*.jpg')) +
                   sorted(Path(images_dir).glob('*.png')))
    print(f"  Found {len(image_paths)} images in {images_dir}")

    all_names: list[str] = []
    all_vecs: list[np.ndarray] = []
    batch_names: list[str] = []
    batch_tensors: list[torch.Tensor] = []

    def flush_batch():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        for name, vec in zip(batch_names, embeddings):
            all_names.append(name)
            all_vecs.append(vec)
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
    print(f"  Processed {len(all_names)}/{len(image_paths)} images")

    if not all_names:
        return [], np.empty((0,), dtype=np.float32)

    matrix = normalize(np.stack(all_vecs), norm='l2').astype(np.float32)
    return all_names, matrix


def save_vectors(names: list[str], matrix: np.ndarray, cfg: dict):
    np.savez_compressed(cfg['npz_out'], **dict(zip(names, matrix)))
    print(f"  Saved NPZ → {cfg['npz_out']}")

    if cfg['json_out']:
        with open(cfg['json_out'], 'w') as f:
            json.dump({n: matrix[i].tolist() for i, n in enumerate(names)}, f)
        print(f"  Saved JSON → {cfg['json_out']}")

    print(f"  Embedding dimension: {matrix.shape[1]}  |  Total: {len(names)}")


# ============================================================
# STEP 2: BUILD FAISS INDICES
# ============================================================

def build_index(matrix: np.ndarray, names: list[str], index_path: str, names_path: str, label: str):
    print(f"  Indexing {label}...")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, index_path)
    np.save(names_path, np.array(names))
    print(f"    {len(names)} vectors → {index_path}")
    return index


def load_index_from_npz(npz_path: str, index_path: str, names_path: str, label: str):
    print(f"  Loading {label} from {npz_path}...")
    data   = np.load(npz_path)
    names  = list(data.files)
    matrix = normalize(np.stack([data[n] for n in names]), norm='l2').astype(np.float32)
    return build_index(matrix, names, index_path, names_path, label), names, matrix


# ============================================================
# STEP 3: MATCH
# ============================================================

def match_and_save(primary_index, primary_names, sec_matrix, sec_names):
    print(f"\n[3/3] Matching secondary against primary (threshold={THRESHOLD})...")
    scores, indices = primary_index.search(sec_matrix, k=1)

    matches, no_match = [], []
    for i, sec_name in enumerate(sec_names):
        sim      = float(scores[i, 0])
        pri_name = primary_names[indices[i, 0]]
        record   = {
            'secondary_image': sec_name,
            'primary_match':   pri_name,
            'similarity':      f"{sim:.4f}",
        }
        (matches if sim >= THRESHOLD else no_match).append(record)

    matches.sort(key=lambda x: x['similarity'], reverse=True)

    print(f"  Above threshold : {len(matches)}")
    print(f"  Below threshold : {len(no_match)}")

    if matches:
        print(f"\n{'Secondary image':<40} {'Primary match':<40} {'Score':>6}")
        print('-' * 88)
        for m in matches:
            print(f"  {m['secondary_image']:<38} {m['primary_match']:<38} {m['similarity']:>6}")
    else:
        print("\nNo matches above threshold.")

    csv_path  = os.path.join(VECTORS_DIR, 'matches_faiss.csv')
    json_path = os.path.join(VECTORS_DIR, 'matches_faiss.json')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['secondary_image', 'primary_match', 'similarity'])
        writer.writeheader()
        writer.writerows(matches)

    with open(json_path, 'w') as f:
        json.dump({'matches': matches, 'no_match': no_match}, f, indent=2)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Vectorize images and match with FAISS')
    parser.add_argument('--skip-vectorize', action='store_true',
                        help='Skip vectorization and use existing NPZ files')
    args = parser.parse_args()

    os.makedirs(VECTORS_DIR, exist_ok=True)

    dataset_vectors: dict[str, tuple[list[str], np.ndarray]] = {}

    if args.skip_vectorize:
        print("[1/3] Skipping vectorization — loading existing NPZ files...")
        for name, cfg in DATASETS.items():
            if not os.path.exists(cfg['npz_out']):
                raise FileNotFoundError(f"NPZ not found: {cfg['npz_out']}. Run without --skip-vectorize first.")
            data   = np.load(cfg['npz_out'])
            names  = list(data.files)
            matrix = normalize(np.stack([data[n] for n in names]), norm='l2').astype(np.float32)
            dataset_vectors[name] = (names, matrix)
            print(f"  Loaded {len(names)} vectors for {name}")
    else:
        print(f"[1/3] Vectorising images with {DINOV2_MODEL} on {DEVICE}...")
        model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL)
        model.eval().to(DEVICE)

        for name, cfg in DATASETS.items():
            print(f"\n  [{name}]")
            names, matrix = vectorize(model, cfg['images_dir'])
            if len(names) == 0:
                raise RuntimeError(f"No images could be embedded for {name}.")
            save_vectors(names, matrix, cfg)
            dataset_vectors[name] = (names, matrix)

        # Free GPU/MPS memory before FAISS
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n[2/3] Building FAISS indices...")
    indices = {}
    for name, cfg in DATASETS.items():
        names, matrix = dataset_vectors[name]
        idx = build_index(matrix, names, cfg['index_out'], cfg['names_out'], name)
        indices[name] = idx

    pri_names, _ = dataset_vectors['primary']
    sec_names, sec_matrix = dataset_vectors['secondary']

    match_and_save(indices['primary'], pri_names, sec_matrix, sec_names)

    print("\nDone.")


if __name__ == '__main__':
    main()
