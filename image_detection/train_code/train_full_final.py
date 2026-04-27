import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import json
import random
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

OUT_DIR = "../Comparison_And_Ablation_Runs/08_Final_AFMBWNetV2_Full"
TARGET_SIZE = 576
EPOCHS = 300

BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 3e-4
WEIGHT_DECAY = 1e-4

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def calculate_iou(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()



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


class AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(inter_ch)

    def forward(self, x_skip, x_gate):
        if x_gate.shape[-2:] != x_skip.shape[-2:]:
            x_gate = F.interpolate(x_gate, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)
        f = F.silu(self.bn(self.theta(x_skip) + self.phi(x_gate)), inplace=True)
        return x_skip * torch.sigmoid(self.psi(f))


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))]
        )
        for r in rates[1:]:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True)
                )
            )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches] + [
            F.interpolate(self.global_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        ]
        return self.proj(torch.cat(feats, dim=1))


class UpAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.gate = AttentionGate(skip_ch, in_ch, max(out_ch // 2, 16))
        self.c = ConvBNAct(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.r = ResBlockECA(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.r(self.c(torch.cat([x, self.gate(skip, x)], dim=1)))


class AFMBWNetV2_Full(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, base, 3, 1, 1), ResBlockECA(base))
        self.d1, self.d2 = Down(base, base * 2), Down(base * 2, base * 4)
        self.mid = nn.Sequential(ResBlockECA(base * 4), ASPP(base * 4, base * 4), ResBlockECA(base * 4))
        self.u1, self.u0 = UpAtt(base * 4, base * 2, base * 2), UpAtt(base * 2, base, base)
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
        main_bce = self.bce(out_main, gt)
        main_dice = self.dice(out_main, gt)

        gt_aux = F.interpolate(gt, size=out_aux.shape[-2:])
        aux_bce = self.bce(out_aux, gt_aux)
        aux_dice = self.dice(out_aux, gt_aux)

        l_main = main_bce + main_dice
        l_aux = aux_bce + aux_dice
        total = l_main + 0.4 * l_aux

        return total, main_bce, main_dice, aux_bce, aux_dice


def train_final_model():
    print(f"\n{'=' * 60}\nStart training: Shape-aware full model (AFMBWNetV2_Shape)\n{'=' * 60}\n")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)

    ds_tr = AFMBWDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, TARGET_SIZE, train=True)
    ds_va = AFMBWDataset(VAL_IMG_DIR, VAL_JSON_DIR, TARGET_SIZE, train=False)

    if len(ds_tr) == 0:
        raise RuntimeError("Training set is empty! Please check the path.")
    print(f"Training set size: {len(ds_tr)} | Validation set size: {len(ds_va)}\n")

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = AFMBWNetV2_Full().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    loss_fn = RegionLoss()

    best_val_loss = float('inf')
    global_start_time = time.time()

    for ep in range(1, EPOCHS + 1):
        # --- Training ---
        train_start_time = time.time()
        model.train()

        tr_total = 0.0
        tr_main_bce = 0.0
        tr_main_dice = 0.0
        tr_aux_bce = 0.0
        tr_aux_dice = 0.0
        tr_iou = 0.0

        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out_main, out_aux = model(x)
            loss, main_bce, main_dice, aux_bce, aux_dice = loss_fn(out_main, out_aux, y)
            loss.backward()
            opt.step()

            tr_total += float(loss.item())
            tr_main_bce += float(main_bce.item())
            tr_main_dice += float(main_dice.item())
            tr_aux_bce += float(aux_bce.item())
            tr_aux_dice += float(aux_dice.item())
            tr_iou += calculate_iou(out_main, y)

        tr_total /= len(dl_tr)
        tr_main_bce /= len(dl_tr)
        tr_main_dice /= len(dl_tr)
        tr_aux_bce /= len(dl_tr)
        tr_aux_dice /= len(dl_tr)
        tr_iou /= len(dl_tr)
        train_time = time.time() - train_start_time

        
        val_start_time = time.time()
        model.eval()

        va_total = 0.0
        va_main_bce = 0.0
        va_main_dice = 0.0
        va_aux_bce = 0.0
        va_aux_dice = 0.0
        va_iou = 0.0

        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                out_main, out_aux = model(x)
                loss, main_bce, main_dice, aux_bce, aux_dice = loss_fn(out_main, out_aux, y)

                va_total += float(loss.item())
                va_main_bce += float(main_bce.item())
                va_main_dice += float(main_dice.item())
                va_aux_bce += float(aux_bce.item())
                va_aux_dice += float(aux_dice.item())
                va_iou += calculate_iou(out_main, y)

        va_total /= len(dl_va)
        va_main_bce /= len(dl_va)
        va_main_dice /= len(dl_va)
        va_aux_bce /= len(dl_va)
        va_aux_dice /= len(dl_va)
        va_iou /= len(dl_va)
        val_time = time.time() - val_start_time

        sch.step(va_total)
        curr_lr = opt.param_groups[0]['lr']

        remaining_epochs = EPOCHS - ep
        eta_seconds = (train_time + val_time) * remaining_epochs
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        log_str = (
            f"[Ep {ep:03d}/{EPOCHS}] "
            f"LR={curr_lr:.2e} | "
            f"Tr_Loss={tr_total:.4f} | Val_Loss={va_total:.4f} | "
            f"Tr_IoU={tr_iou:.4f} | Val_IoU={va_iou:.4f} | "
            f"Time:{train_time + val_time:.1f}s | ETA: {eta_str}"
        )
        print(log_str)

        if va_total < best_val_loss:
            best_val_loss = va_total
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pth"))
            print(f"  Save best model (Val Loss updated to: {best_val_loss:.4f})")

    total_time = time.time() - global_start_time
    print(f"\nAll training completed! Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f" Model saved at: {OUT_DIR}")


if __name__ == "__main__":
    train_final_model()
