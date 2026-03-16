import os
import json
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import normalize
import torch
import torchvision.transforms as T

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
PRIMARY_NPZ          = os.path.join(BASE_DIR, 'data', 'vectors', 'image_vectors.npz')
SECONDARY_DIR        = os.path.join(BASE_DIR, 'data', 'auction_images_secondary')
OUTPUT_DIR           = os.path.join(BASE_DIR, 'data', 'vectors')
SIMILARITY_THRESHOLD = 0.80

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

def load_image(path):
    try:
        return transform(Image.open(path).convert('RGB'))
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None

# ============================================================
# 1. LOAD & NORMALISE PRIMARY VECTORS
# ============================================================

print("Loading primary vectors...")
data           = np.load(PRIMARY_NPZ)
primary_names  = list(data.files)
primary_matrix = normalize(np.stack([data[n] for n in primary_names]), norm='l2')
N, D           = primary_matrix.shape
print(f"  {N} primary images, {D}-dim embeddings")

# ============================================================
# 2. VECTORISE SECONDARY IMAGES
# ============================================================

print(f"\nLoading DINOv2 ({DINOV2_MODEL}) on {DEVICE}...")
model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL)
model.eval().to(DEVICE)

image_paths = (sorted(Path(SECONDARY_DIR).glob('*.jpeg')) +
               sorted(Path(SECONDARY_DIR).glob('*.jpg')) +
               sorted(Path(SECONDARY_DIR).glob('*.png')))
print(f"Vectorising {len(image_paths)} secondary images...")

sec_names, sec_vectors = [], []
batch_names, batch_tensors = [], []

def flush_batch():
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
        flush_batch()
        print(f"  {i + 1}/{len(image_paths)}")

flush_batch()
print(f"  {len(image_paths)}/{len(image_paths)}")

sec_matrix = normalize(np.stack(sec_vectors), norm='l2')   # (M, D)

# ============================================================
# 3. COSINE SIMILARITY: secondary × primary  →  (M, N)
# ============================================================

print(f"\nMatching (cosine similarity ≥ {SIMILARITY_THRESHOLD:.0%})...")
sim_matrix = sec_matrix @ primary_matrix.T    # (M, N)

matches, no_match = [], []
for i, sec_name in enumerate(sec_names):
    best_idx = int(np.argmax(sim_matrix[i]))
    cos_sim  = float(sim_matrix[i, best_idx])

    record = {
        'secondary_image': sec_name,
        'primary_match':   primary_names[best_idx],
        'similarity':      round(cos_sim, 4),
    }
    if cos_sim >= SIMILARITY_THRESHOLD:
        matches.append(record)
    else:
        no_match.append({**record, 'primary_match': primary_names[best_idx]})

matches.sort(key=lambda x: x['similarity'], reverse=True)
print(f"  Matches found:   {len(matches)}")
print(f"  Below threshold: {len(no_match)}")

# ============================================================
# 4. SAVE RESULTS
# ============================================================

json_path = os.path.join(OUTPUT_DIR, 'matches.json')
with open(json_path, 'w') as f:
    json.dump({'matches': matches, 'no_match': no_match}, f, indent=2)

csv_path = os.path.join(OUTPUT_DIR, 'matches.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['secondary_image', 'primary_match', 'similarity'])
    writer.writeheader()
    writer.writerows(matches)

print(f"\nResults saved:")
print(f"  {json_path}")
print(f"  {csv_path}")
print("Done.")
