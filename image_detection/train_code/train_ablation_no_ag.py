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
OUT_DIR = "../Comparison_And_Ablation_Runs/07_Ablation_NoAG"
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
        mask_w, mask_b = np.zeros((self.ts, self.ts), dtype=np.float32), np.zeros((self.ts, self.ts), dtype=np.float32)
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

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ECALayer(nn.Module):
    def __init__(self, ch, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
    
    def forward(self, x):
        y = self.avg(x).squeeze(-1).transpose(-1, -2)
        return x * torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)

class ResBlockECA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.c2 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch))
        self.eca = ECALayer(ch)
    
    def forward(self, x):
        return self.eca(F.silu(self.c2(self.c1(x)) + x, inplace=True))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.d = ConvBNAct(in_ch, out_ch, 3, 2, 1)
        self.r = ResBlockECA(out_ch)
    
    def forward(self, x):
        return self.r(self.d(x))

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))])
        for r in rates[1:]:
            self.branches.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True)))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))
    
    def forward(self, x):
        feats = [b(x) for b in self.branches] + [F.interpolate(self.global_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)]
        return self.proj(torch.cat(feats, dim=1))

class UpNoAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.c = ConvBNAct(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.r = ResBlockECA(out_ch)
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.r(self.c(torch.cat([x, skip], dim=1)))

class Model_NoAG(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, base, 3, 1, 1), ResBlockECA(base))
        self.d1, self.d2 = Down(base, base * 2), Down(base * 2, base * 4)
        self.mid = nn.Sequential(ResBlockECA(base * 4), ASPP(base * 4, base * 4), ResBlockECA(base * 4))
        self.u1, self.u0 = UpNoAtt(base * 4, base * 2, base * 2), UpNoAtt(base * 2, base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.aux_head = nn.Conv2d(base * 4, out_ch, 1)
    
    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        m = self.mid(x2)
        return self.head(self.u0(self.u1(m, x1), x0)), self.aux_head(m)

class RegionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    
    def forward(self, out_main, out_aux, gt):
        bce_main = self.bce(out_main, gt)
        dice_main = self.dice(out_main, gt)
        gt_aux = F.interpolate(gt, size=out_aux.shape[-2:])
        bce_aux = self.bce(out_aux, gt_aux)
        dice_aux = self.dice(out_aux, gt_aux)
        total_bce = bce_main + 0.4 * bce_aux
        total_dice = dice_main + 0.4 * dice_aux
        return total_bce + total_dice, total_bce, total_dice

def calculate_iou(pred_logits, gt_masks):
    pred_probs = torch.sigmoid(pred_logits)
    pred_bool = (pred_probs > 0.5).float()
    intersection = (pred_bool * gt_masks).sum(dim=(2, 3))
    union = pred_bool.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def train_ablation_no_ag():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("\n Warning: No 5090 GPU detected! Running at slow speed!\n")
    else:
        print(f"\n Detected heavy firepower: {torch.cuda.get_device_name(0)}! Starting [No AG Gate Ablation Experiment]...\n")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    ds_tr = AFMBWDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, TARGET_SIZE, train=True)
    ds_va = AFMBWDataset(VAL_IMG_DIR, VAL_JSON_DIR, TARGET_SIZE, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = Model_NoAG().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    loss_fn = RegionLoss()
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
                out_main, out_aux = model(x)
                loss, bce, dice = loss_fn(out_main, out_aux, y)
            
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
                    out_main, out_aux = model(x)
                    loss, bce, dice = loss_fn(out_main, out_aux, y)
                
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
        
        log_str = (f"[NoAG - Ep {ep:03d}/{EPOCHS}] LR={curr_lr:.2e} | "
                  f"Tr[L:{tr_loss:.3f}, IoU:{tr_iou:.3f}] | "
                  f"Val[L:{va_loss:.3f}, IoU:{va_iou:.3f}] | "
                  f"Time:{ep_time:.1f}s | ETA: {eta_str}")
        print(log_str)
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pth"))
    
    total_time = time.time() - global_start_time
    print(f"\nAblation experiment (No AG) completed! Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

if __name__ == "__main__":
    train_ablation_no_ag()