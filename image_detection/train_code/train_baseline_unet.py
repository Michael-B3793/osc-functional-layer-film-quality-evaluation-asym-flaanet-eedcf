import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
import json
import random
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

TRAIN_IMG_DIR = r"../dataset_400/train_images"
TRAIN_JSON_DIR = r"../dataset_400/train_jsons"
VAL_IMG_DIR = r"../dataset_400/val_images"
VAL_JSON_DIR = r"../dataset_400/val_jsons"
OUT_DIR = "../Comparison_And_Ablation_Runs/01_Baseline_UNet"
TARGET_SIZE = 576
EPOCHS = 300
BATCH_SIZE = 8
NUM_WORKERS = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def read_height_proxy(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed: {img_path}")
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.float32)

def robust_norm01(x, p_low=1.0, p_high=99.0):
    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    return ((np.clip(x, lo, hi) - lo) / (hi - lo + 1e-6)).astype(np.float32)

def detrend_by_gaussian(z01, sigma=18.0):
    out = z01 - cv2.GaussianBlur(z01, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return (out - out.min()) / (out.max() - out.min() + 1e-6)

def clahe01(z01):
    u8 = np.clip(z01 * 255.0, 0, 255).astype(np.uint8)
    return (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0)

def local_rms(z01, k=9):
    mean, mean2 = cv2.blur(z01, (k, k)), cv2.blur(z01 * z01, (k, k))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0) + 1e-6).astype(np.float32)

def grad_mag(z01):
    gx, gy = cv2.Sobel(z01, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(z01, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)

class AFMBWDataset(Dataset):
    def __init__(self, img_dir, json_dir, target_size=576, train=True):
        self.img_dir, self.json_dir = img_dir, json_dir
        self.ts, self.train = int(target_size), bool(train)
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    
    def __len__(self):
        return len(self.files)
    
    def _load_masks(self, fn, orig_w, orig_h):
        jp = os.path.join(self.json_dir, os.path.splitext(fn)[0] + ".json")
        mask_w = np.zeros((self.ts, self.ts), dtype=np.float32)
        mask_b = np.zeros((self.ts, self.ts), dtype=np.float32)
        if os.path.exists(jp):
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            sx, sy = self.ts / float(orig_w), self.ts / float(orig_h)
            for shape in data.get("shapes", []):
                label = shape["label"].lower()
                pts = (np.array(shape["points"]) * [sx, sy]).astype(np.int32)
                if label == 'white':
                    cv2.fillPoly(mask_w, [pts], 1.0)
                elif label == 'black':
                    cv2.fillPoly(mask_b, [pts], 1.0)
        return mask_w, mask_b
    
    def __getitem__(self, idx):
        fn = self.files[idx]
        img_path = os.path.join(self.img_dir, fn)
        z = read_height_proxy(img_path)
        orig_h, orig_w = z.shape[:2]
        z01 = cv2.resize(robust_norm01(z), (self.ts, self.ts), interpolation=cv2.INTER_LINEAR)
        z01 = clahe01(detrend_by_gaussian(z01, sigma=18.0))
        ch2, ch3 = local_rms(z01, k=9), grad_mag(z01)
        mask_w, mask_b = self._load_masks(fn, orig_w, orig_h)
        
        if self.train and np.random.rand() < 0.5:
            op = random.choice(["hflip", "vflip", "rot90"])
            if op == "hflip":
                z01, ch2, ch3 = np.fliplr(z01).copy(), np.fliplr(ch2).copy(), np.fliplr(ch3).copy()
                mask_w, mask_b = np.fliplr(mask_w).copy(), np.fliplr(mask_b).copy()
            elif op == "vflip":
                z01, ch2, ch3 = np.flipud(z01).copy(), np.flipud(ch2).copy(), np.flipud(ch3).copy()
                mask_w, mask_b = np.flipud(mask_w).copy(), np.flipud(mask_b).copy()
            elif op == "rot90":
                z01, ch2, ch3 = np.rot90(z01, 1).copy(), np.rot90(ch2, 1).copy(), np.rot90(ch3, 1).copy()
                mask_w, mask_b = np.rot90(mask_w, 1).copy(), np.rot90(mask_b, 1).copy()
        
        x = np.stack([z01, ch2, ch3], axis=0).astype(np.float32)
        y = np.stack([mask_w, mask_b], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

class SingleRegionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    
    def forward(self, out_main, gt):
        bce_main = self.bce(out_main, gt)
        dice_main = self.dice(out_main, gt)
        total_loss = bce_main + dice_main
        return total_loss, bce_main, dice_main

def calculate_iou(pred_logits, gt_masks):
    pred_probs = torch.sigmoid(pred_logits)
    pred_bool = (pred_probs > 0.5).float()
    intersection = (pred_bool * gt_masks).sum(dim=(2, 3))
    union = pred_bool.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def train_baseline_unet():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("\n Warning: No GPU detected!")
    else:
        print(f"\n Detected GPU: {torch.cuda.get_device_name(0)}, starting U-Net baseline high-speed training...\n")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    ds_tr = AFMBWDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, TARGET_SIZE, train=True)
    ds_va = AFMBWDataset(VAL_IMG_DIR, VAL_JSON_DIR, TARGET_SIZE, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    loss_fn = SingleRegionLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = 1e9
    global_start_time = time.time()
    
    for ep in range(1, EPOCHS + 1):
        ep_start_time = time.time()
        model.train()
        tr_loss, tr_bce, tr_dice, tr_iou = 0, 0, 0, 0
        
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out_main = model(x)
                loss, bce, dice = loss_fn(out_main, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            tr_loss += loss.item()
            tr_bce += bce.item()
            tr_dice += dice.item()
            tr_iou += calculate_iou(out_main, y)
        
        model.eval()
        va_loss, va_bce, va_dice, va_iou = 0, 0, 0, 0
        
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out_main = model(x)
                    loss, bce, dice = loss_fn(out_main, y)
                
                va_loss += loss.item()
                va_bce += bce.item()
                va_dice += dice.item()
                va_iou += calculate_iou(out_main, y)
        
        tr_loss, tr_bce, tr_dice, tr_iou = [v / len(dl_tr) for v in (tr_loss, tr_bce, tr_dice, tr_iou)]
        va_loss, va_bce, va_dice, va_iou = [v / len(dl_va) for v in (va_loss, va_bce, va_dice, va_iou)]
        
        curr_lr = opt.param_groups[0]['lr']
        sch.step(va_loss)
        
        ep_time = time.time() - ep_start_time
        eta_seconds = ep_time * (EPOCHS - ep)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        log_str = (f"[Ep {ep:03d}/{EPOCHS}] LR={curr_lr:.2e} | "
                  f"Tr[L:{tr_loss:.3f}, IoU:{tr_iou:.3f}] | "
                  f"Val[L:{va_loss:.3f}, IoU:{va_iou:.3f}] | "
                  f"Time:{ep_time:.1f}s | ETA: {eta_str}")
        print(log_str)
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pth"))
    
    total_time = time.time() - global_start_time
    print(f"\nU-Net baseline training completed! Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

if __name__ == "__main__":
    train_baseline_unet()