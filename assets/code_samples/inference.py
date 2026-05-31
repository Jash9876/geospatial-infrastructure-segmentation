import os
import cv2
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

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 5

# ======================
# COLORS
# ======================
COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
}

# ======================
# MODEL
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
# TTA
# ======================
def tta_predict(model, img):
    preds = []

    preds.append(model(img))

    img_h = torch.flip(img, dims=[3])
    preds.append(torch.flip(model(img_h), dims=[3]))

    img_v = torch.flip(img, dims=[2])
    preds.append(torch.flip(model(img_v), dims=[2]))

    img_hv = torch.flip(img, dims=[2, 3])
    preds.append(torch.flip(model(img_hv), dims=[2, 3]))

    return torch.mean(torch.stack(preds), dim=0)


# ======================
# COLOR MASK
# ======================
def decode_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLORS.items():
        color_mask[mask == cls] = color

    return color_mask


# ======================
# METRICS
# ======================
def compute_metrics(pred, mask):
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

    return pixel_acc, np.nanmean(ious), ious


# ======================
# RUN INFERENCE
# ======================
imgs = sorted(glob(os.path.join(IMG_DIR, "*.npy")))

for i, path in enumerate(tqdm(imgs[:50])):  # limit for demo

    img = np.load(path)
    orig = img.copy()

    inp = preprocess(img).to(DEVICE)

    with torch.no_grad():
        out = tta_predict(model, inp)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    # ======================
    # VISUALS
    # ======================
    color_mask = decode_mask(pred)

    overlay = cv2.addWeighted(orig, 0.6, color_mask, 0.4, 0)

    # SAVE
    cv2.imwrite(f"{OUT_DIR}/mask_{i}.png", color_mask)
    cv2.imwrite(f"{OUT_DIR}/overlay_{i}.png", overlay)

    # ======================
    # METRICS (if GT exists)
    # ======================
    if os.path.exists(MASK_DIR):
        gt = np.load(os.path.join(MASK_DIR, os.path.basename(path)))

        acc, miou, ious = compute_metrics(pred, gt)

        print(f"\nImage {i}")
        print(f"Pixel Acc: {acc:.4f}, mIoU: {miou:.4f}")

        names = ["BG", "Road", "Roof", "Water", "Utility"]
        for j in range(NUM_CLASSES):
            print(f"{names[j]}: {ious[j]:.4f}")

print("\n✅ INFERENCE COMPLETE")