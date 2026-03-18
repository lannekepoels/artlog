"""
Annoy workflow: vectorize secondary images, build Annoy index, match against primary.

Prerequisites: run vectorize_images.py --dataset primary first.

Outputs (data/vectors/):
  primary.ann / primary_names.npy
  matches_annoy.csv
  matches_annoy.json
"""

import os
import gc
import csv
import json
import numpy as np
import torch
import torchvision.transforms as T
from annoy import AnnoyIndex
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import normalize

# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD    = 0.80
N_TREES      = 50    # more trees = better accuracy, slower build
SEARCH_K     = -1    # -1 = auto (n_trees * k); increase for better recall

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VECTORS_DIR   = os.path.join(BASE_DIR, 'data', 'vectors')
SECONDARY_DIR = os.path.join(BASE_DIR, 'data', 'auction_images_secondary')
PRIMARY_NPZ   = os.path.join(VECTORS_DIR, 'image_vectors.npz')
PRIMARY_ANN   = os.path.join(VECTORS_DIR, 'primary.ann')
PRIMARY_NAMES = os.path.join(VECTORS_DIR, 'primary_names_annoy.npy')

DINOV2_MODEL = 'dinov2_vitb14'
BATCH_SIZE   = 16
DEVICE       = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================
# STEP 1: VECTORIZE SECONDARY IMAGES (in-memory)
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


print(f"[1/3] Vectorising secondary images with {DINOV2_MODEL} on {DEVICE}...")
model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL)
model.eval().to(DEVICE)

image_paths = (sorted(Path(SECONDARY_DIR).glob('*.jpeg')) +
               sorted(Path(SECONDARY_DIR).glob('*.jpg')) +
               sorted(Path(SECONDARY_DIR).glob('*.png')))
print(f"  Found {len(image_paths)} images")

sec_names: list[str] = []
sec_vectors: list[np.ndarray] = []
batch_names: list[str] = []
batch_tensors: list[torch.Tensor] = []


def flush():
    if not batch_tensors:
        return
    batch = torch.stack(batch_tensors).to(DEVICE)
    with torch.no_grad():
        emb = model(batch).cpu().numpy()
    for name, vec in zip(batch_names, emb):
        sec_names.append(name)
        sec_vectors.append(vec)
    batch_names.clear()
    batch_tensors.clear()


for i, path in enumerate(image_paths):
    t = load_image(str(path))
    if t is None:
        continue
    batch_names.append(path.name)
    batch_tensors.append(t)
    if len(batch_tensors) >= BATCH_SIZE:
        flush()
        print(f"  {i + 1}/{len(image_paths)}")

flush()
print(f"  {len(sec_names)}/{len(image_paths)} images embedded")

sec_matrix = normalize(np.stack(sec_vectors), norm='l2').astype(np.float32)

# Free GPU/MPS memory before building index
del model, batch_tensors, sec_vectors
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ============================================================
# STEP 2: BUILD ANNOY INDEX FROM PRIMARY VECTORS
# ============================================================

os.makedirs(VECTORS_DIR, exist_ok=True)

print("\n[2/3] Building Annoy index for primary vectors...")
data          = np.load(PRIMARY_NPZ)
primary_names = list(data.files)
primary_matrix = normalize(np.stack([data[n] for n in primary_names]), norm='l2').astype(np.float32)

N, D = primary_matrix.shape
print(f"  {N} primary images, {D}-dim embeddings")

# Annoy uses angular distance, which is equivalent to cosine similarity
# on L2-normalised vectors: angular distance = arccos(cosine_similarity)
ann_index = AnnoyIndex(D, metric='angular')
for idx, vec in enumerate(primary_matrix):
    ann_index.add_item(idx, vec)

print(f"  Building index with {N_TREES} trees...")
ann_index.build(N_TREES)
ann_index.save(PRIMARY_ANN)
np.save(PRIMARY_NAMES, np.array(primary_names))
print(f"  Saved → {PRIMARY_ANN}")

# ============================================================
# STEP 3: MATCH SECONDARY AGAINST PRIMARY
# ============================================================

print(f"\n[3/3] Matching secondary against primary (threshold={THRESHOLD})...")

matches, no_match = [], []
# NOTE: Annoy's get_nns_by_vector is broken on Python 3.13 (returns 1 result regardless of n).
# Fall back to numpy dot-product (exact cosine similarity on L2-normalised vectors).
# With only 303 primary images this is fast and correct.
cos_matrix = primary_matrix @ sec_matrix.T  # (N_primary, N_secondary)

for i, sec_name in enumerate(sec_names):
    best_idx = int(np.argmax(cos_matrix[:, i]))
    cos_sim  = float(cos_matrix[best_idx, i])
    pri_name = primary_names[best_idx]

    record = {
        'secondary_image': sec_name,
        'primary_match':   pri_name,
        'similarity':      f"{cos_sim:.4f}",
    }
    (matches if cos_sim >= THRESHOLD else no_match).append(record)

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

# ============================================================
# SAVE RESULTS
# ============================================================

csv_path  = os.path.join(VECTORS_DIR, 'matches_annoy.csv')
json_path = os.path.join(VECTORS_DIR, 'matches_annoy.json')

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['secondary_image', 'primary_match', 'similarity'])
    writer.writeheader()
    writer.writerows(matches)

with open(json_path, 'w') as f:
    json.dump({'matches': matches, 'no_match': no_match}, f, indent=2)

print(f"\nSaved → {csv_path}")
print(f"Saved → {json_path}")
print("\nDone.")
