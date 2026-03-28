"""
FAISS workflow: vectorize secondary images, build indices, match against primary.

Prerequisites: run vectorize_images.py --dataset primary first.

Outputs (data/vectors/):
  primary.faiss / primary_names.npy
  secondary.faiss / secondary_names.npy
  matches_faiss.csv
  matches_faiss.json
"""

import os
import gc
import csv
import json
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
SECONDARY_DIR   = os.path.join(BASE_DIR, 'data', 'dataset_A_images_secondary')
PRIMARY_NPZ     = os.path.join(VECTORS_DIR, 'image_vectors.npz')
PRIMARY_INDEX   = os.path.join(VECTORS_DIR, 'primary.faiss')
PRIMARY_NAMES   = os.path.join(VECTORS_DIR, 'primary_names.npy')
SECONDARY_INDEX = os.path.join(VECTORS_DIR, 'secondary.faiss')
SECONDARY_NAMES = os.path.join(VECTORS_DIR, 'secondary_names.npy')

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

# Free GPU/MPS memory before loading FAISS
del model, batch_tensors, sec_vectors
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ============================================================
# STEP 2: BUILD FAISS INDICES
# ============================================================

os.makedirs(VECTORS_DIR, exist_ok=True)


def build_index_from_npz(npz_path: str, index_path: str, names_path: str, label: str):
    print(f"  Indexing {label}...")
    data   = np.load(npz_path)
    names  = list(data.files)
    matrix = normalize(np.stack([data[n] for n in names]), norm='l2').astype(np.float32)
    index  = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, index_path)
    np.save(names_path, np.array(names))
    print(f"    {len(names)} vectors → {index_path}")
    return index, names, matrix


def build_index_from_matrix(matrix: np.ndarray, names: list, index_path: str, names_path: str, label: str):
    print(f"  Indexing {label}...")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, index_path)
    np.save(names_path, np.array(names))
    print(f"    {len(names)} vectors → {index_path}")
    return index


print("\n[2/3] Building FAISS indices...")
primary_index, primary_names, _ = build_index_from_npz(PRIMARY_NPZ, PRIMARY_INDEX, PRIMARY_NAMES, 'primary')
secondary_index = build_index_from_matrix(sec_matrix, sec_names, SECONDARY_INDEX, SECONDARY_NAMES, 'secondary')

# ============================================================
# STEP 3: MATCH SECONDARY AGAINST PRIMARY
# ============================================================

print(f"\n[3/3] Matching secondary against primary (threshold={THRESHOLD})...")
scores, indices = primary_index.search(sec_matrix, k=1)   # (M, 1)

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

# ============================================================
# SAVE RESULTS
# ============================================================

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
print("\nDone.")
