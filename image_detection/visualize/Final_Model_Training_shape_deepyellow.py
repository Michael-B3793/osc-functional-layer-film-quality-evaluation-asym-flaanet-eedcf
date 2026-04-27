import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset_400/text_images")
    TEST_JSON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset_400/val_jsons")
    MODEL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "Final_Model_Training_shape/best.pth")
    OUTPUT_IMG_DIR = os.path.join(SCRIPT_DIR, "Final_Model_Training_shape_deepyellow")
    OUTPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "Final_Model_Training_shape_deepyellow/Detection_Metrics.csv")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_SIZE = 576
    MASK_THRESH = 0.5


    BG_DEEP_YELLOW = (0, 109, 180)  
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    THICKNESS = -1  


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
                    nn.SiLU(inplace=True),
                )
            )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
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


def read_height_proxy(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
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
    mean, mean2 = cv2.blur(z01, (9, 9)), cv2.blur(z01 * z01, (9, 9))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0) + 1e-6).astype(np.float32)


def grad_mag(z01):
    gx, gy = cv2.Sobel(z01, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(z01, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def process_contours_to_metrics(contours, d_type, img_name, source_name):
    metrics = []
    d_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3.0:
            continue
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(cnt[0][0][0])
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(cnt[0][0][1])
        x, y, w, h = cv2.boundingRect(cnt)
        metrics.append(
            {
                "Image": img_name,
                "Source": source_name,
                "Type": d_type,
                "ID": d_id,
                "CX": cx,
                "CY": cy,
                "X_min": x,
                "Y_min": y,
                "W": w,
                "H": h,
                "Area": round(area, 2),
            }
        )
        d_id += 1
    return metrics


def make_deepyellow_canvas(size):
    h, w = size, size
    canvas = np.full((h, w, 3), Config.BG_DEEP_YELLOW, dtype=np.uint8)
    return canvas


if __name__ == "__main__":
    os.makedirs(Config.OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.OUTPUT_CSV_PATH), exist_ok=True)
    print("Loading region segmentation model...")
    model = AFMBWNetV2_Full()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    target_image = "4.tiff"
    if not os.path.exists(os.path.join(Config.TEST_IMG_DIR, target_image)):
        target_image = "4.tif"

    image_files = [target_image] if os.path.exists(os.path.join(Config.TEST_IMG_DIR, target_image)) else []
    all_metrics = []

    if not image_files:
        print(f"Target image not found: {target_image}")
    else:
        for img_name in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(Config.TEST_IMG_DIR, img_name)
            json_path = os.path.join(Config.TEST_JSON_DIR, os.path.splitext(img_name)[0] + ".json")

            z = read_height_proxy(img_path)
            if z is None:
                continue

            orig_h, orig_w = z.shape[:2]
            z01 = cv2.resize(robust_norm01(z), (Config.TARGET_SIZE, Config.TARGET_SIZE))
            z01_c = clahe01(detrend_by_gaussian(z01))
            ch2, ch3 = local_rms(z01_c), grad_mag(z01_c)
            x = torch.from_numpy(np.stack([z01_c, ch2, ch3])).unsqueeze(0).to(Config.DEVICE)

            canvas_gt = make_deepyellow_canvas(Config.TARGET_SIZE)
            canvas_pred = make_deepyellow_canvas(Config.TARGET_SIZE)

            c_w_gt, c_b_gt = [], []
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sx, sy = Config.TARGET_SIZE / float(orig_w), Config.TARGET_SIZE / float(orig_h)
                    for shape in data.get("shapes", []):
                        pts = (np.array(shape["points"]) * [sx, sy]).astype(np.int32)
                        if shape["label"].lower() == "white":
                            c_w_gt.append(pts)
                        elif shape["label"].lower() == "black":
                            c_b_gt.append(pts)

                all_metrics.extend(process_contours_to_metrics(c_w_gt, "White", img_name, "GT"))
                all_metrics.extend(process_contours_to_metrics(c_b_gt, "Black", img_name, "GT"))

                cv2.drawContours(canvas_gt, c_w_gt, -1, Config.COLOR_WHITE, Config.THICKNESS)
                cv2.drawContours(canvas_gt, c_b_gt, -1, Config.COLOR_BLACK, Config.THICKNESS)

            with torch.no_grad():
                prob = torch.sigmoid(model(x)[0]).cpu().numpy()
                mask_w_pred = (prob[0, 0] > Config.MASK_THRESH).astype(np.uint8)
                mask_b_pred = (prob[0, 1] > Config.MASK_THRESH).astype(np.uint8)

                c_w_pred, _ = cv2.findContours(mask_w_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c_b_pred, _ = cv2.findContours(mask_b_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                all_metrics.extend(process_contours_to_metrics(c_w_pred, "White", img_name, "Pred"))
                all_metrics.extend(process_contours_to_metrics(c_b_pred, "Black", img_name, "Pred"))

                cv2.drawContours(canvas_pred, c_w_pred, -1, Config.COLOR_WHITE, Config.THICKNESS)
                cv2.drawContours(canvas_pred, c_b_pred, -1, Config.COLOR_BLACK, Config.THICKNESS)

            base_name = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(Config.OUTPUT_IMG_DIR, f"{base_name}_GT.png"), canvas_gt)
            cv2.imwrite(os.path.join(Config.OUTPUT_IMG_DIR, f"{base_name}_Pred.png"), canvas_pred)

        if all_metrics:
            df = pd.DataFrame(all_metrics).sort_values(by=["Image", "Source", "Type"])
            df.to_csv(Config.OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"\nEvaluation completed! Recorded {len(df)} region data entries.")

