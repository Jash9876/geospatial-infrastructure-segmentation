import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import segmentation_models_pytorch as smp

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_model.pth"
IMG_DIR = r"C:\images"
MASK_DIR = r"C:\masks"

NUM_CLASSES = 5

# ======================
# LOAD MODEL
# ======================
model = smp.DeepLabV3Plus(
    encoder_name="mit_b4",
    encoder_weights=None,
    classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# ======================
# NORMALIZATION
# ======================
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def preprocess(img):
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).float().unsqueeze(0)


# ======================
# METRICS
# ======================
def compute_metrics(pred, mask):
    pred = pred.cpu().numpy()
    mask = mask.cpu().numpy()

    pixel_acc = (pred == mask).sum() / mask.size

    ious = []
    for cls in range(NUM_CLASSES):
        p = (pred == cls)
        m = (mask == cls)

        inter = (p & m).sum()
        union = (p | m).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)

    return pixel_acc, ious


# ======================
# RUN
# ======================
img_paths = sorted(glob(os.path.join(IMG_DIR, "*.npy")))
mask_paths = sorted(glob(os.path.join(MASK_DIR, "*.npy")))

total_acc = 0
all_ious = []
valid_count = 0

for img_p, mask_p in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):

    img = np.load(img_p)
    mask = np.load(mask_p)

    # 🔥 SKIP PURE BACKGROUND (IMPORTANT)
    unique = np.unique(mask)
    if len(unique) == 1 and unique[0] == 0:
        continue

    inp = preprocess(img).to(DEVICE)

    with torch.no_grad():
        out = model(inp)
        pred = torch.argmax(out, dim=1).squeeze()

    acc, ious = compute_metrics(pred, torch.tensor(mask))

    total_acc += acc
    all_ious.append(ious)
    valid_count += 1


# ======================
# FINAL RESULTS
# ======================
mean_acc = total_acc / valid_count
mean_ious = np.nanmean(all_ious, axis=0)
miou = np.nanmean(mean_ious)

print("\n📊 FILTERED METRICS (MATCHES TRAINING)")
print(f"Valid Samples: {valid_count}")
print(f"Pixel Accuracy: {mean_acc:.4f}")
print(f"Mean IoU: {miou:.4f}")

names = ["Background", "Road", "Roof", "Water", "Utility"]
print("\nClass-wise IoU:")
for i in range(NUM_CLASSES):
    print(f"{names[i]}: {mean_ious[i]:.4f}")