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

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class Config:
    TRAIN_IMG_DIR = r"image_detection/dataset_400/train_images"
    TRAIN_JSON_DIR = r"image_detection/dataset_400/train_jsons"
    VAL_IMG_DIR = r"image_detection/dataset_400/val_images"
    VAL_JSON_DIR = r"image_detection/dataset_400/val_jsons"

    OUT_ROOT_DIR = "image_detection/All_Models_Training_Logs"
    TARGET_SIZE = 576
    EPOCHS = 300

    BATCH_SIZE = 8
    NUM_WORKERS = 0
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42

    WEIGHT_PATH_SAF = "model.safetensors"
    WEIGHT_PATH_PTH = "mit_b3.pth"


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


def robust_norm01(x):
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
    return ((np.clip(x, lo, hi) - lo) / (hi - lo + 1e-6)).astype(np.float32)


def detrend_by_gaussian(z01):
    out = z01 - cv2.GaussianBlur(z01, (0, 0), sigmaX=18.0, sigmaY=18.0)
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def clahe01(z01):
    u8 = np.clip(z01 * 255.0, 0, 255).astype(np.uint8)
    return (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0)


def local_rms(z01):
    mean = cv2.blur(z01, (9, 9))
    mean2 = cv2.blur(z01 * z01, (9, 9))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0) + 1e-6).astype(np.float32)


def grad_mag(z01):
    gx = cv2.Sobel(z01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z01, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def calculate_iou(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


class AFMRegionDataset(Dataset):
    def __init__(self, img_dir, json_dir, mode='all', train=True):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.ts = Config.TARGET_SIZE
        self.train = bool(train)
        self.mode = mode
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
        z01_c = clahe01(detrend_by_gaussian(z01))
        ch2 = local_rms(z01_c)
        ch3 = grad_mag(z01_c)
        mask_w, mask_b = self._load_masks(fn, orig_w, orig_h)

        if self.train and np.random.rand() < 0.5:
            op = random.choice(["hflip", "vflip", "rot90"])
            if op == "hflip":
                z01_c = np.fliplr(z01_c).copy()
                ch2 = np.fliplr(ch2).copy()
                ch3 = np.fliplr(ch3).copy()
                mask_w = np.fliplr(mask_w).copy()
                mask_b = np.fliplr(mask_b).copy()
            elif op == "vflip":
                z01_c = np.flipud(z01_c).copy()
                ch2 = np.flipud(ch2).copy()
                ch3 = np.flipud(ch3).copy()
                mask_w = np.flipud(mask_w).copy()
                mask_b = np.flipud(mask_b).copy()
            elif op == "rot90":
                z01_c = np.rot90(z01_c, 1).copy()
                ch2 = np.rot90(ch2, 1).copy()
                ch3 = np.rot90(ch3, 1).copy()
                mask_w = np.rot90(mask_w, 1).copy()
                mask_b = np.rot90(mask_b, 1).copy()

        if self.mode == 'all':
            x = np.stack([z01_c, ch2, ch3], axis=0).astype(np.float32)
            y = np.stack([mask_w, mask_b], axis=0).astype(np.float32)
        elif self.mode == 'white_expert':
            x = np.stack([z01_c, ch2, ch3], axis=0).astype(np.float32)
            y = np.expand_dims(mask_w, axis=0).astype(np.float32)
        elif self.mode == 'black_expert':
            x = np.stack([1.0 - z01_c, ch2, ch3], axis=0).astype(np.float32)
            y = np.expand_dims(mask_b, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

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
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))])
        for r in rates[1:]:
            self.branches.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True)))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.global_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
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


class UpNoAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.c = ConvBNAct(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.r = ResBlockECA(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.r(self.c(torch.cat([x, skip], dim=1)))


class AFMBWNetV2_Full(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, base, 3, 1, 1), ResBlockECA(base))
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.mid = nn.Sequential(ResBlockECA(base * 4), ASPP(base * 4, base * 4), ResBlockECA(base * 4))
        self.u1 = UpAtt(base * 4, base * 2, base * 2)
        self.u0 = UpAtt(base * 2, base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.aux_head = nn.Conv2d(base * 4, out_ch, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        m = self.mid(x2)
        return self.head(self.u0(self.u1(m, x1), x0)), self.aux_head(m)


class Model_NoASPP(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, base, 3, 1, 1), ResBlockECA(base))
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.mid = nn.Sequential(ResBlockECA(base * 4), ResBlockECA(base * 4), ResBlockECA(base * 4))
        self.u1 = UpAtt(base * 4, base * 2, base * 2)
        self.u0 = UpAtt(base * 2, base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.aux_head = nn.Conv2d(base * 4, out_ch, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        m = self.mid(x2)
        return self.head(self.u0(self.u1(m, x1), x0)), self.aux_head(m)


class Model_NoAG(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=2):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, base, 3, 1, 1), ResBlockECA(base))
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.mid = nn.Sequential(ResBlockECA(base * 4), ASPP(base * 4, base * 4), ResBlockECA(base * 4))
        self.u1 = UpNoAtt(base * 4, base * 2, base * 2)
        self.u0 = UpNoAtt(base * 2, base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        self.aux_head = nn.Conv2d(base * 4, out_ch, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        m = self.mid(x2)
        return self.head(self.u0(self.u1(m, x1), x0)), self.aux_head(m)


def build_mit_b3_smp(classes):
    model = smp.Unet(encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=classes, decoder_attention_type="scse")
    weight_path = Config.WEIGHT_PATH_SAF if os.path.exists(Config.WEIGHT_PATH_SAF) else Config.WEIGHT_PATH_PTH
    if os.path.exists(weight_path):
        try:
            if weight_path.endswith(".safetensors") and HAS_SAFETENSORS:
                state_dict = load_file(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model_dict = model.encoder.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.encoder.load_state_dict(model_dict)
            print(f"MiT-B3 encoder weights loaded successfully: {len(pretrained_dict)} parameters matched")
        except Exception as e:
            print(f"MiT-B3 weight loading failed, continuing with current initialization: {e}")
    else:
        print("No MiT-B3 external weight file found, continuing with current initialization")
    return model


class UnifiedRegionLoss(nn.Module):
    def __init__(self, has_aux=False):
        super().__init__()
        self.has_aux = has_aux
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='multilabel', from_logits=True)

    def forward(self, outputs, gt):
        if self.has_aux:
            out_main, out_aux = outputs
            main_bce = self.bce(out_main, gt)
            main_dice = self.dice(out_main, gt)
            gt_aux = F.interpolate(gt, size=out_aux.shape[-2:], mode="nearest")
            aux_bce = self.bce(out_aux, gt_aux)
            aux_dice = self.dice(out_aux, gt_aux)
            total_loss = (main_bce + main_dice) + 0.4 * (aux_bce + aux_dice)
        else:
            main_bce = self.bce(outputs, gt)
            main_dice = self.dice(outputs, gt)
            aux_bce = None
            aux_dice = None
            total_loss = main_bce + main_dice
        return {
            "total_loss": total_loss,
            "main_bce": main_bce.detach(),
            "main_dice": main_dice.detach(),
            "aux_bce": None if aux_bce is None else aux_bce.detach(),
            "aux_dice": None if aux_dice is None else aux_dice.detach(),
        }


def init_csv(csv_path, headers):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(headers)


def append_csv(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)


def value_or_zero(x):
    if x is None:
        return 0.0
    return float(x.item() if torch.is_tensor(x) else x)


def run_experiment(exp_name, model, mode, has_aux):
    print(f"\n{'=' * 60}\nStart training and recording: {exp_name}\n{'=' * 60}")
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running device: {device}")

    save_dir = os.path.join(Config.OUT_ROOT_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "training_log.txt")
    train_csv_file = os.path.join(save_dir, "train_epoch_metrics.csv")
    val_csv_file = os.path.join(save_dir, "val_epoch_metrics.csv")

    csv_headers = ["Epoch", "LR", "Total_Loss", "Main_BCE", "Main_Dice", "Aux_BCE", "Aux_Dice", "Main_IoU", "Phase_Time(s)"]
    init_csv(train_csv_file, csv_headers)
    init_csv(val_csv_file, csv_headers)

    ds_tr = AFMRegionDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_JSON_DIR, mode=mode, train=True)
    ds_va = AFMRegionDataset(Config.VAL_IMG_DIR, Config.VAL_JSON_DIR, mode=mode, train=False)

    if len(ds_tr) == 0:
        raise RuntimeError("Training set is empty! Please check the path.")
    print(f"Training set size: {len(ds_tr)} | Validation set size: {len(ds_va)}\n")

    dl_tr = DataLoader(ds_tr, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    dl_va = DataLoader(ds_va, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    if isinstance(model, nn.Module):
        model = model.to(device)
    else:
        model = model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    loss_fn = UnifiedRegionLoss(has_aux=has_aux)

    global_start_time = time.time()

    for ep in range(1, Config.EPOCHS + 1):
        train_start_time = time.time()
        model.train()

        tr_total = tr_main_bce = tr_main_dice = tr_aux_bce = tr_aux_dice = tr_iou = 0.0

        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            outputs = model(x)
            loss_dict = loss_fn(outputs, y)
            loss = loss_dict["total_loss"]
            loss.backward()
            opt.step()

            out_main = outputs[0] if has_aux else outputs
            tr_total += loss.item()
            tr_main_bce += value_or_zero(loss_dict["main_bce"])
            tr_main_dice += value_or_zero(loss_dict["main_dice"])
            tr_aux_bce += value_or_zero(loss_dict["aux_bce"])
            tr_aux_dice += value_or_zero(loss_dict["aux_dice"])
            tr_iou += calculate_iou(out_main, y)

        train_time = time.time() - train_start_time

        val_start_time = time.time()
        model.eval()

        va_total = va_main_bce = va_main_dice = va_aux_bce = va_aux_dice = va_iou = 0.0

        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss_dict = loss_fn(outputs, y)

                out_main = outputs[0] if has_aux else outputs
                va_total += loss_dict["total_loss"].item()
                va_main_bce += value_or_zero(loss_dict["main_bce"])
                va_main_dice += value_or_zero(loss_dict["main_dice"])
                va_aux_bce += value_or_zero(loss_dict["aux_bce"])
                va_aux_dice += value_or_zero(loss_dict["aux_dice"])
                va_iou += calculate_iou(out_main, y)

        val_time = time.time() - val_start_time

        num_tr = max(1, len(dl_tr))
        num_va = max(1, len(dl_va))

        tr_total /= num_tr
        tr_main_bce /= num_tr
        tr_main_dice /= num_tr
        tr_aux_bce /= num_tr
        tr_aux_dice /= num_tr
        tr_iou /= num_tr

        va_total /= num_va
        va_main_bce /= num_va
        va_main_dice /= num_va
        va_aux_bce /= num_va
        va_aux_dice /= num_va
        va_iou /= num_va

        sch.step(va_total)
        curr_lr = opt.param_groups[0]["lr"]

        epoch_time = train_time + val_time
        remaining_epochs = Config.EPOCHS - ep
        eta_seconds = epoch_time * remaining_epochs
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        log_str = (
            f"[Ep {ep:03d}/{Config.EPOCHS}] "
            f"LR={curr_lr:.2e} | "
            f"Tr_Loss={tr_total:.4f} | Val_Loss={va_total:.4f} | "
            f"Tr_IoU={tr_iou:.4f} | Val_IoU={va_iou:.4f} | "
            f"Time:{epoch_time:.1f}s | ETA: {eta_str}"
        )
        print(log_str)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        append_csv(train_csv_file, [ep, curr_lr, tr_total, tr_main_bce, tr_main_dice, tr_aux_bce, tr_aux_dice, tr_iou, train_time])
        append_csv(val_csv_file, [ep, curr_lr, va_total, va_main_bce, va_main_dice, va_aux_bce, va_aux_dice, va_iou, val_time])

    total_time = time.time() - global_start_time
    print(f" {exp_name} training completed!")
    print(f"{exp_name} total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n")


if __name__ == "__main__":
    os.makedirs(Config.OUT_ROOT_DIR, exist_ok=True)

    experiments = [
        ("01_Baseline_UNet", smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2), 'all', False),
        ("02_Structure_UNetPP", smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2), 'all', False),
        ("03_Attention_TransUNet", build_mit_b3_smp(classes=2), 'all', False),
        ("04_Strategy_Dual_WhiteExpert", build_mit_b3_smp(classes=1), 'white_expert', False),
        ("05_Strategy_Dual_BlackExpert", build_mit_b3_smp(classes=1), 'black_expert', False),
        ("06_Ablation_NoASPP", Model_NoASPP(in_ch=3, base=32, out_ch=2), 'all', True),
        ("07_Ablation_NoAG", Model_NoAG(in_ch=3, base=32, out_ch=2), 'all', True),
      
        ("08_Final_AFMBWNetV2_Full", lambda: AFMBWNetV2_Full(in_ch=3, base=32, out_ch=2), 'all', True)
    ]

    print("================================================================")
    print(f" About to execute {len(experiments)} models with full metric monitoring training")
    print("================================================================\n")

    for exp_name, model, mode, has_aux in experiments:
        try:
            run_experiment(exp_name, model, mode, has_aux)
        except Exception as e:
            print(f" Experiment {exp_name} encountered a fatal error: {e}")
            import traceback
            traceback.print_exc()

    print(" Congratulations! All 8 models have been trained, and train_epoch_metrics.csv and val_epoch_metrics.csv have been generated for each model, ready for plotting!")
