import os
import cv2
import time
import random
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as A

torch.backends.cudnn.benchmark = True


# ======================
# CONFIG (CHANGE STAGE)
# ======================
STAGE = 1

CONFIG = {
    1: {"size": 512, "bs": 4, "lr": 1e-4, "load": None},
    2: {"size": 768, "bs": 2, "lr": 5e-5, "load": "best_stage1.pth"},
    3: {"size": 1024, "bs": 1, "lr": 1e-5, "load": "best_stage2.pth"},
}

cfg = CONFIG[STAGE]

IMG_SIZE = cfg["size"]
BATCH_SIZE = cfg["bs"]
LR = cfg["lr"]
LOAD_PATH = cfg["load"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5

TRAIN_IMG = r"C:\images"
TRAIN_MASK = r"C:\masks"
VAL_IMG = r"C:\val\images"
VAL_MASK = r"C:\val\masks"

CHECKPOINT = f"checkpoint_stage{STAGE}.pth"
BEST_MODEL = f"best_stage{STAGE}.pth"

CLASS_NAMES = ["Background", "Road", "Roof", "Water", "Utility"]


# ======================
# DATASET
# ======================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(glob(os.path.join(img_dir, "*.tif")))
        self.masks = sorted(glob(os.path.join(mask_dir, "*.tif")))

        if len(self.imgs) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i])
        mask = cv2.imread(self.masks[i], 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Hybrid crop / resize
        if random.random() < 0.5 and h >= IMG_SIZE and w >= IMG_SIZE:
            x = random.randint(0, w - IMG_SIZE)
            y = random.randint(0, h - IMG_SIZE)
            img = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            mask = mask[y:y+IMG_SIZE, x:x+IMG_SIZE]
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Multi-scale
        scale = random.choice([0.75, 1.0, 1.25])
        new_size = int(IMG_SIZE * scale)

        img = cv2.resize(img, (new_size, new_size))
        mask = cv2.resize(mask, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9,1.1), rotate=(-15,15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.CLAHE(p=0.2),
        ])

        data = aug(image=img, mask=mask)
        img, mask = data["image"], data["mask"]

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img).float(), torch.tensor(mask).long()


# ======================
# TTA (VALIDATION)
# ======================
def tta_predict(model, img):
    pred1 = model(img)

    flipped = torch.flip(img, dims=[3])
    pred2 = torch.flip(model(flipped), dims=[3])

    return (pred1 + pred2) / 2


# ======================
# POST PROCESS (HOOK)
# ======================
def post_process(mask):
    return mask  # placeholder


# ======================
# LOSS
# ======================
ce = nn.CrossEntropyLoss(
    weight=torch.tensor([0.2, 3.0, 1.2, 2.0, 4.0]).to(DEVICE)
)

dice = smp.losses.DiceLoss(mode="multiclass")


class BoundaryLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()

        edge_x = torch.abs(pred[:,:,:,:-1] - pred[:,:,:,1:])
        edge_y = torch.abs(pred[:,:,:-1,:] - pred[:,:,1:,:])

        target_x = torch.abs(target[:,:,:,:-1] - target[:,:,:,1:])
        target_y = torch.abs(target[:,:,:-1,:] - target[:,:,1:,:])

        return torch.mean((edge_x - target_x)**2) + torch.mean((edge_y - target_y)**2)


boundary = BoundaryLoss()


def total_loss(pred, target):
    return 0.4*dice(pred,target) + 0.4*ce(pred,target) + 0.2*boundary(pred,target)


# ======================
# MAIN
# ======================
def main():

    print("🚀 Training Stage:", STAGE)

    global_start = time.time()

    train_loader = DataLoader(
        SegDataset(TRAIN_IMG, TRAIN_MASK),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        SegDataset(VAL_IMG, VAL_MASK),
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    model = smp.DeepLabV3Plus(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        classes=NUM_CLASSES,
        decoder_channels=256
    ).to(DEVICE)

    if LOAD_PATH and os.path.exists(LOAD_PATH):
        print("🔄 Loading previous stage...")
        model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=6
    )

    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best, no_improve = 0, 0, 0

    if os.path.exists(CHECKPOINT):
        print("🔄 Resuming training...")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt['best']
        no_improve = ckpt['no_improve']

    ACCUM_STEPS = 2

    while True:

        epoch = start_epoch
        start_epoch += 1

        ep_start = time.time()
        model.train()
        train_loss = 0

        optimizer.zero_grad()

        for i, (img, mask) in enumerate(tqdm(train_loader)):
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = total_loss(out, mask) / ACCUM_STEPS

            scaler.scale(loss).backward()

            # ✅ Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        # ======================
        # VALIDATION (WITH TTA)
        # ======================
        model.eval()
        class_iou_sum = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)

                out = tta_predict(model, img)

                pred = torch.argmax(out, 1)
                pred = post_process(pred)

                for c in range(NUM_CLASSES):
                    inter = ((pred == c) & (mask == c)).sum().item()
                    union = ((pred == c) | (mask == c)).sum().item()
                    iou = 1 if union == 0 else inter / union
                    class_iou_sum[c] += iou

        class_iou_avg = class_iou_sum / len(val_loader)
        val_score = np.mean(class_iou_avg)

        scheduler.step(val_score)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {train_loss/len(train_loader):.4f}")
        print(f"mIoU: {val_score:.4f}")

        weakest = np.argmin(class_iou_avg)
        print(f"⚠️ Weakest Class: {CLASS_NAMES[weakest]}")

        if val_score > best:
            best = val_score
            no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL)
            print("🔥 BEST SAVED")
        else:
            no_improve += 1

        print(f"📊 Best: {best:.4f} | No improve: {no_improve}")

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best': best,
            'no_improve': no_improve
        }, CHECKPOINT)

        epoch_time = time.time() - ep_start
        elapsed = time.time() - global_start
        avg_epoch = elapsed / (epoch + 1)
        eta = avg_epoch * max(0, 30 - epoch)

        print(f"⏱ Epoch Time: {int(epoch_time)}s")
        print(f"⏳ ETA Remaining: {int(eta//60)} min")

        if epoch > 20 and no_improve >= 10:
            print("🛑 Converged. Stop Stage.")
            break


if __name__ == "__main__":
    main()