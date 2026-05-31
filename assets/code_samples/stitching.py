import os
import numpy as np
from glob import glob
from tqdm import tqdm

PRED_DIR = r"C:\predictions"
SAVE_DIR = r"C:\stitched"

os.makedirs(SAVE_DIR, exist_ok=True)

TILE_SIZE = 1024  # 🔥 IMPORTANT

tiles = glob(os.path.join(PRED_DIR, "*.npy"))

village_dict = {}

# ======================
# GROUP TILES
# ======================
for path in tiles:
    name = os.path.basename(path).replace(".npy", "")
    parts = name.split("_")

    try:
        row = int(parts[-2])
        col = int(parts[-1])
    except:
        continue

    village = "_".join(parts[:-2])

    if village not in village_dict:
        village_dict[village] = []

    village_dict[village].append((row, col, path))

# ======================
# STITCH FIXED GRID
# ======================
for village, data in village_dict.items():

    print(f"\n🧩 Stitching {village}")

    max_row = max(d[0] for d in data)
    max_col = max(d[1] for d in data)

    full = np.zeros(
        ((max_row+1)*TILE_SIZE, (max_col+1)*TILE_SIZE),
        dtype=np.uint8
    )

    for row, col, path in tqdm(data):
        tile = np.load(path)

        # pad small tiles
        h, w = tile.shape
        padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        padded[:h, :w] = tile

        full[
            row*TILE_SIZE:(row+1)*TILE_SIZE,
            col*TILE_SIZE:(col+1)*TILE_SIZE
        ] = padded

    np.save(os.path.join(SAVE_DIR, f"{village}.npy"), full)

print("\n✅ Stitching done")