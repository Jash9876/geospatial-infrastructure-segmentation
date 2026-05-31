import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

# ======================
# DATASET
# ======================
class RoofDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).long()

        return img, mask

# ======================
# VISUAL FUNCTION
# ======================
def visualize(model, dataset, device):
    model.eval()

    img, mask = dataset[np.random.randint(len(dataset))]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    img_np = img.permute(1,2,0).numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img_np)

    plt.subplot(1,3,2)
    plt.title("GT")
    plt.imshow(mask)

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred)

    plt.show()

# ======================
# MAIN TRAIN FUNCTION
# ======================
def main():

    # ======================
    # CONFIG
    # ======================
    IMG_DIR = r"F:\IIT\roof_type_dataset\images"
    MASK_DIR = r"F:\IIT\roof_type_dataset\masks"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 4
    EPOCHS = 12
    NUM_CLASSES = 4
    LR = 5e-5

    CHECKPOINT = r"F:\IIT\roof_checkpoint.pth"
    BEST_MODEL = r"F:\IIT\roof_best_resnet.pth"

    print(f"🚀 Using device: {DEVICE}")

    # ======================
    # DATA
    # ======================
    dataset = RoofDataset(IMG_DIR, MASK_DIR)

    print("📊 Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # ✅ FIXED
        pin_memory=True
    )

    # ======================
    # LOSS
    # ======================
    weights = torch.tensor([0.01, 1.0, 10.0, 3.0]).to(DEVICE)

    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    def loss_fn(pred, target):
        return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    # ======================
    # MODEL
    # ======================
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ======================
    # RESUME
    # ======================
    start_epoch = 0
    best_loss = float("inf")

    if os.path.exists(CHECKPOINT):
        print("♻️ Resuming from checkpoint...")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["best_loss"]

    # ======================
    # TRAIN LOOP
    # ======================
    for epoch in range(start_epoch, EPOCHS):

        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

        model.train()
        total_loss = 0

        for img, mask in tqdm(loader):

            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            out = model(img)
            loss = loss_fn(out, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"📉 Avg Loss: {avg_loss:.4f}")

        # ======================
        # SAVE BEST MODEL
        # ======================
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL)
            print("🔥 BEST MODEL SAVED")

        # ======================
        # SAVE CHECKPOINT
        # ======================
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss
        }, CHECKPOINT)

        # ======================
        # VISUAL CHECK
        # ======================
        visualize(model, dataset, DEVICE)

    print("\n✅ TRAINING COMPLETE")


# ======================
# ENTRY POINT (CRITICAL FIX)
# ======================
if __name__ == "__main__":
    main()