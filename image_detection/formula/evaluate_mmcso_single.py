import os
import cv2
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_IMAGE_PATH = os.path.join(SCRIPT_DIR, "../dataset_400/text_images/4.tif")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "../Comparison_And_Ablation_Runs/Final_Model_Training_shape/best.pth")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_SIZE = 576
    MASK_THRESH = 0.5
    W = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    TOL_AREA = (576 * 576) * 0.05
    TOL_AVG_AREA = 500.0
    TOL_COUNT = 50.0

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
        feats = [b(x) for b in self.branches] + [F.interpolate(self.global_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)]
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

def robust_norm01(x):
    return ((np.clip(x, np.percentile(x, 1), np.percentile(x, 99)) - np.percentile(x, 1)) / (np.percentile(x, 99) - np.percentile(x, 1) + 1e-6)).astype(np.float32)

def detrend_by_gaussian(z01):
    return (z01 - cv2.GaussianBlur(z01, (0, 0), sigmaX=18.0, sigmaY=18.0))

def clahe01(z01):
    return (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(np.clip((z01-z01.min())/(z01.max()-z01.min()+1e-6)*255.0, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0)

def local_rms(z01):
    mean, mean2 = cv2.blur(z01, (9,9)), cv2.blur(z01*z01, (9,9))
    return np.sqrt(np.maximum(mean2 - mean*mean, 0.0) + 1e-6).astype(np.float32)

def grad_mag(z01):
    gx, gy = cv2.Sobel(z01, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(z01, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)

def extract_and_calculate_mmcso(prob_w, prob_b, z_raw):
    print("\n" + "="*50)
    print("Algorithm 2: MMCSO (Macro-Micro Coupled SNR Operator) Calculation Report")
    print("="*50)
    z_min, z_max = z_raw.min(), z_raw.max()
    rq_color_bar = (z_raw - z_min) / (z_max - z_min + 1e-6)
    mu_global = float(np.mean(rq_color_bar))
    
    mask_w = (prob_w > Config.MASK_THRESH).astype(np.uint8)
    mask_b = (prob_b > Config.MASK_THRESH).astype(np.uint8)
    c_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features_list = []
    print("Extracting microscopic features (building signal)...")
    for idx, cnt in enumerate(c_w + c_b):
        area = cv2.contourArea(cnt)
        if area < 3.0:
            continue
        
        M = cv2.moments(cnt)
        cx, cy = M['m10']/M['m00'] if M['m00']!=0 else cnt[0][0][0], M['m01']/M['m00'] if M['m00']!=0 else cnt[0][0][1]
        
        pts = cnt.squeeze(1)
        if pts.ndim == 1:
            pts = np.expand_dims(pts, 0)
        
        distances = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
        ratio = np.min(distances) / max(np.max(distances), 1e-6)
        
        mask = np.zeros_like(rq_color_bar, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        mu_k = cv2.mean(rq_color_bar, mask=mask)[0]
        
        features_list.append({"area": area, "ratio": ratio, "mu_k": mu_k})
    
    N = len(features_list)
    P6 = 2.0 * abs(mu_global - 0.5)
    
    if N == 0:
        print(" No defects detected, theoretical maximum value will be output.")
        return
    
    A_def = sum([f["area"] for f in features_list])
    A_avg = A_def / N
    P1, P2, P4 = min(A_def/Config.TOL_AREA, 1.0), min(A_avg/Config.TOL_AVG_AREA, 1.0), min(N/Config.TOL_COUNT, 1.0)
    P3 = sum([1.0 - abs(2*f["mu_k"] - 1.0) for f in features_list]) / N
    P5 = sum([1.0 - f["ratio"] for f in features_list]) / N
    
    print(f"\n 1. Building effective feature signal (Signal, larger is better):")
    sig_color = Config.W[2]*(1-P3)
    sig_morph = Config.W[4]*(1-P5)
    sig_bg = Config.W[5]*(1-P6)
    Signal = sig_color + sig_morph + sig_bg
    print(f"  -> Color contrast gain (W={Config.W[2]}): {sig_color:.4f}")
    print(f"  -> Morphology compactness gain (W={Config.W[4]}): {sig_morph:.4f}")
    print(f"  -> Global background health gain (W={Config.W[5]}): {sig_bg:.4f}")
    print(f"  Total effective signal Signal = {Signal:.4f}")
    
    print(f"\n 2. Building macro pollution noise (Noise, smaller is better):")
    noise_area = Config.W[0]*P1
    noise_avg = Config.W[1]*P2
    noise_count = Config.W[3]*P4
    Noise = noise_area + noise_avg + noise_count
    print(f"  -> Total area pollution (W={Config.W[0]}): {noise_area:.4f}")
    print(f"  -> Average area pollution (W={Config.W[1]}): {noise_avg:.4f}")
    print(f"  -> Density pollution (W={Config.W[3]}): {noise_count:.4f}")
    print(f"  Total background noise Noise = {Noise:.4f}")
    
    MMCSO = 10.0 * math.log10(1.0 + Signal / (Noise + 1e-6))
    print(f"\nFinal MMCSO SNR = 10 * log10(1 + Sig/Noise) = {MMCSO:.2f} dB")
    print("(Interpretation: Larger MMCSO indicates that effective morphology signals in the image overwhelm pollution noise, resulting in excellent image quality)")
    print("="*50)

if __name__ == "__main__":
    print(f"Script directory: {Config.SCRIPT_DIR}")
    print(f"Test image path: {Config.TEST_IMAGE_PATH}")
    print(f"Model path: {Config.MODEL_PATH}")
    print(f"Test image exists: {os.path.exists(Config.TEST_IMAGE_PATH)}")
    print(f"Model exists: {os.path.exists(Config.MODEL_PATH)}")
    
    if not os.path.exists(Config.TEST_IMAGE_PATH):
        print(" Test image not found")
        exit()
    
    model = AFMBWNetV2_Full().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()
    
    img = cv2.imread(Config.TEST_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = 0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]
    
    z_raw = cv2.resize(img.astype(np.float32), (Config.TARGET_SIZE, Config.TARGET_SIZE))
    z01_c = clahe01(detrend_by_gaussian(robust_norm01(z_raw.copy())))
    ch2, ch3 = local_rms(z01_c), grad_mag(z01_c)
    
    x = torch.from_numpy(np.stack([z01_c, ch2, ch3])).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)[0][0]).cpu().numpy()
        extract_and_calculate_mmcso(prob[0], prob[1], z_raw)
