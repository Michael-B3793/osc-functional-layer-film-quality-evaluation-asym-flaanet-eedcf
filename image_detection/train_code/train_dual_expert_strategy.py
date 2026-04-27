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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

TRAIN_IMG_DIR = r"../dataset_400/train_images"
TRAIN_JSON_DIR = r"../dataset_400/train_jsons"
VAL_IMG_DIR = r"../dataset_400/val_images"
VAL_JSON_DIR = r"../dataset_400/val_jsons"
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

class AFMBWDatasetExpert(Dataset):
    def __init__(self, img_dir, json_dir, target_size=576, train=True):
        self.img_dir, self.json_dir = img_dir, json_dir
        self.ts, self.train = int(target_size), bool(train)
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    
    def __len__(self):
        return len(self.files)
    
    def _load_mask_monoclass(self, fn, orig_w, orig_h, target_label):
        jp = os.path.join(self.json_dir, os.path.splitext(fn)[0] + ".json")
        mask = np.zeros((self.ts, self.ts), dtype=np.float32)
        if os.path.exists(jp):
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            sx, sy = self.ts / float(orig_w), self.ts / float(orig_h)
            for shape in data.get("shapes", []):
                if shape["label"].lower() == target_label.lower():
                    pts = (np.array(shape["points"]) * [sx, sy]).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1.0)
        return mask
    
    def __getitem__(self, idx):
        fn = self.files[idx]
        img_path = os.path.join(self.img_dir, fn)
        z = read_height_proxy(img_path)
        orig_h, orig_w = z.shape[:2]
        z01 = robust_norm01(z)
        z01 = cv2.resize(z01, (self.ts, self.ts))
        z01_enh = clahe01(z01)
        ch2, ch3 = local_rms(z01_enh, k=9), grad_mag(z01_enh)
        if self.train and np.random.rand() < 0.5:
            op = random.choice(["hflip", "vflip"])
            if op == "hflip":
                z01_enh, ch2, ch3 = np.fliplr(z01_enh).copy(), np.fliplr(ch2).copy(), np.fliplr(ch3).copy()
            else:
                z01_enh, ch2, ch3 = np.flipud(z01_enh).copy(), np.flipud(ch2).copy(), np.flipud(ch3).copy()
        x = np.stack([z01_enh, ch2, ch3], axis=0).astype(np.float32)
        return x, fn, orig_w, orig_h

class SingleChannelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    def forward(self, out_main, gt):
        bce_main = self.bce(out_main, gt)
        dice_main = self.dice(out_main, gt)
        total_loss = bce_main + dice_main
        return total_loss, bce_main, dice_main

def calculate_iou_expert(pred_logits, gt_masks):
    pred_probs = torch.sigmoid(pred_logits)
    pred_bool = (pred_probs > 0.5).float()
    intersection = (pred_bool * gt_masks).sum(dim=(2, 3))
    union = pred_bool.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def train_expert_model(expert_type):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================== AFM BW Expert Strategy Validation ====================")
    print(f"Current expert type: [ {expert_type.upper()} Protrusion/Depression ]")
    
    if device.type != 'cuda':
        print(" Warning: No GPU detected, running at slow speed!")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f" Detected GPU: {gpu_name}, maximizing Tensor Core performance...")
    
    OUT_DIR = f"../Comparison_And_Ablation_Runs/04_Strategy_Dual_WhiteExpert" if expert_type == "white" else "../Comparison_And_Ablation_Runs/05_Strategy_Dual_BlackExpert"
    print(f"Output directory: {OUT_DIR}\n")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    class CustomDataset(Dataset):
        def __init__(self, base_dataset, expert_type, json_dir, target_size):
            self.base_dataset = base_dataset
            self.expert_type = expert_type
            self.json_dir = json_dir
            self.target_size = target_size
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            x, fn, orig_w, orig_h = self.base_dataset[idx]
            jp = os.path.join(self.json_dir, os.path.splitext(fn)[0] + ".json")
            mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
            if os.path.exists(jp):
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sx, sy = self.target_size / float(orig_w), self.target_size / float(orig_h)
                for shape in data.get("shapes", []):
                    if shape["label"].lower() == self.expert_type.lower():
                        pts = (np.array(shape["points"]) * [sx, sy]).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1.0)
            y = np.expand_dims(mask, axis=0).astype(np.float32)
            return x, y
    
    base_tr = AFMBWDatasetExpert(TRAIN_IMG_DIR, TRAIN_JSON_DIR, TARGET_SIZE, train=True)
    base_va = AFMBWDatasetExpert(VAL_IMG_DIR, VAL_JSON_DIR, TARGET_SIZE, train=False)
    
    ds_tr = CustomDataset(base_tr, expert_type, TRAIN_JSON_DIR, TARGET_SIZE)
    ds_va = CustomDataset(base_va, expert_type, VAL_JSON_DIR, TARGET_SIZE)
    
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print("Building single expert specialized network (Mit-B3 + scSE)...")
    model = smp.Unet(encoder_name="mit_b3", encoder_weights="imagenet", in_channels=3, classes=1, decoder_attention_type="scse").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    loss_fn = SingleChannelLoss()
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
                logits = model(x)
                loss, bce, dice = loss_fn(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            tr_loss += loss.item()
            tr_bce += bce.item()
            tr_dice += dice.item()
            tr_iou += calculate_iou_expert(logits, y)
        
        model.eval()
        va_loss, va_bce, va_dice, va_iou = 0, 0, 0, 0
        
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
                    loss, bce, dice = loss_fn(logits, y)
                
                va_loss += loss.item()
                va_bce += bce.item()
                va_dice += dice.item()
                va_iou += calculate_iou_expert(logits, y)
        
        tr_loss, tr_bce, tr_dice, tr_iou = [v / len(dl_tr) for v in (tr_loss, tr_bce, tr_dice, tr_iou)]
        va_loss, va_bce, va_dice, va_iou = [v / len(dl_va) for v in (va_loss, va_bce, va_dice, va_iou)]
        
        curr_lr = opt.param_groups[0]['lr']
        sch.step(va_loss)
        
        ep_time = time.time() - ep_start_time
        eta_seconds = ep_time * (EPOCHS - ep)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        log_str = (f"[DualExp_{expert_type.upper()} - Ep {ep:03d}/{EPOCHS}] LR={curr_lr:.2e} | "
                  f"Tr[L:{tr_loss:.3f}, IoU:{tr_iou:.3f}] | "
                  f"Val[L:{va_loss:.3f}, IoU:{va_iou:.3f}] | "
                  f"Time:{ep_time:.1f}s | ETA: {eta_str}")
        print(log_str)
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pth"))
    
    total_time = time.time() - global_start_time
    print(f"\nTraining completed! {expert_type.capitalize()} expert model total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

def main():
    print("Starting dual expert model training...")
    print("This will train both white and black expert models sequentially.")
    
    print("\n" + "="*80)
    print("Training White Expert Model")
    print("="*80)
    train_expert_model("white")
    
    print("\n" + "="*80)
    print("Training Black Expert Model")
    print("="*80)
    train_expert_model("black")
    
    print("\n" + "="*80)
    print("Both expert models have been trained successfully!")
    print("="*80)

if __name__ == "__main__":
    main()