"""Train a small 3-class (solder=0, IMC=1, Cu=2) segmentation model on the
24 hand-labeled images in training_data/, using patch sampling + augmentation
since the labeled set is small. Pretrained encoder (transfer learning) per
the plan: fine-tune, don't train from scratch, given ~24 images.
"""
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(r"C:\Users\82109\IMC_Thickness_Tool\training_data")
IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"
CKPT_PATH = Path(r"C:\Users\82109\IMC_Thickness_Tool\imc_segmenter.pt")

# Held out for a rough generalization check -- one per alloy where possible.
VAL_NAMES = {"0_500_27", "50_200_13", "80_1000_9", "100_200_8"}

CROP = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def imread_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


class IMCDataset(Dataset):
    def __init__(self, names, train=True, crops_per_image=40):
        self.items = []
        for name in names:
            img = imread_gray(IMG_DIR / f"{name}.png")
            mask = imread_gray(MASK_DIR / f"{name}.png")
            imc_ys, imc_xs = np.where(mask == 1)  # precomputed once, not per-sample
            self.items.append((name, img, mask, imc_ys, imc_xs))
        self.train = train
        self.crops_per_image = crops_per_image
        self.length = len(self.items) * (crops_per_image if train else 4)

    def __len__(self):
        return self.length

    def _sample_crop(self, img, mask, imc_ys, imc_xs):
        h, w = img.shape
        # bias the crop center toward the IMC band so most patches actually
        # contain the interface, not just deep bulk Cu or deep bulk solder
        if imc_ys.size and random.random() < 0.85:
            i = random.randrange(imc_ys.size)
            cy, cx = int(imc_ys[i]), int(imc_xs[i])
            cy += random.randint(-60, 60)
            cx += random.randint(-60, 60)
        else:
            cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y0 = min(max(cy - CROP // 2, 0), max(h - CROP, 0))
        x0 = min(max(cx - CROP // 2, 0), max(w - CROP, 0))
        y1, x1 = min(y0 + CROP, h), min(x0 + CROP, w)
        img_c = img[y0:y1, x0:x1]
        mask_c = mask[y0:y1, x0:x1]
        if img_c.shape[0] != CROP or img_c.shape[1] != CROP:
            img_c = cv2.copyMakeBorder(img_c, 0, CROP - img_c.shape[0], 0, CROP - img_c.shape[1],
                                        cv2.BORDER_REFLECT)
            mask_c = cv2.copyMakeBorder(mask_c, 0, CROP - mask_c.shape[0], 0, CROP - mask_c.shape[1],
                                         cv2.BORDER_REFLECT)
        return img_c, mask_c

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        if random.random() < 0.3:
            angle = random.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (CROP, CROP), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (CROP, CROP), borderMode=cv2.BORDER_REFLECT,
                                   flags=cv2.INTER_NEAREST)
        img = img.astype(np.float32)
        if random.random() < 0.7:
            img = img * random.uniform(0.8, 1.2) + random.uniform(-15, 15)
        if random.random() < 0.5:
            img = img + np.random.normal(0, random.uniform(2, 8), img.shape)
        img = np.clip(img, 0, 255)
        return img, mask

    def __getitem__(self, idx):
        name, img, mask, imc_ys, imc_xs = self.items[idx % len(self.items)]
        img_c, mask_c = self._sample_crop(img, mask, imc_ys, imc_xs)
        if self.train:
            img_c, mask_c = self._augment(img_c, mask_c)
        img_t = torch.from_numpy(img_c.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask_c.astype(np.int64))
        return img_t, mask_t


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Compact U-Net, trained from scratch but small enough (and the dataset
    augmented enough via dense patch sampling) to fit in 3GB VRAM and avoid
    gross overfitting on 24 source images. No internet-downloaded pretrained
    encoder used (kept dependency-free / offline-safe)."""
    def __init__(self, n_classes=3, base=24):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def main():
    all_names = sorted(p.stem for p in IMG_DIR.glob("*.png"))
    train_names = [n for n in all_names if n not in VAL_NAMES]
    val_names = [n for n in all_names if n in VAL_NAMES]
    print(f"train images: {len(train_names)}, val images: {len(val_names)}", flush=True)
    print("device:", DEVICE, flush=True)

    train_ds = IMCDataset(train_names, train=True, crops_per_image=60)
    val_ds = IMCDataset(val_names, train=False, crops_per_image=8)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = SmallUNet(n_classes=3).to(DEVICE)
    # class weights: IMC is the minority class and the one we care most about
    class_weights = torch.tensor([1.0, 2.5, 1.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 25
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val = float("inf")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)
        sched.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        imc_iou_inter, imc_iou_union = 0, 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)
                val_loss += criterion(pred, mask).item() * img.size(0)
                pred_cls = pred.argmax(dim=1)
                correct += (pred_cls == mask).sum().item()
                total += mask.numel()
                p1, m1 = (pred_cls == 1), (mask == 1)
                imc_iou_inter += (p1 & m1).sum().item()
                imc_iou_union += (p1 | m1).sum().item()
        val_loss /= len(val_ds)
        acc = correct / total
        imc_iou = imc_iou_inter / max(imc_iou_union, 1)
        print(f"epoch {epoch+1:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_acc {acc:.3f}  val_IMC_IoU {imc_iou:.3f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, CKPT_PATH)
            print(f"  -> new best, checkpoint saved", flush=True)

    print("best val_loss:", best_val, "-> saved to", CKPT_PATH, flush=True)


if __name__ == "__main__":
    main()
