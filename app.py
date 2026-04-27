import os
import cv2
import torch
import math
import time
import shutil
import numpy as np
import pandas as pd
import gradio as gr
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F

class Config:
    MODEL_PATH = r"./image_detection/Final_Model_Training_shape/best.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_SIZE = 576
    MASK_THRESH = 0.5
    W1, W2, W3, W4, W5, W6 = 0.35, 0.25, 0.15, 0.10, 0.10, 0.05
    TOL_AREA = (576 * 576) * 0.05
    TOL_AVG_AREA = 500.0
    TOL_COUNT = 50.0
    COLOR_WHITE = (255, 0, 0)
    COLOR_BLACK = (0, 0, 255)
    THICKNESS = 2
    RECORD_DIR = "System_Records"

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

def calculate_multidimensional_metrics(prob_w, prob_b, z_raw):
    z_min, z_max = z_raw.min(), z_raw.max()
    rq_color_bar = (z_raw - z_min) / (z_max - z_min + 1e-6)
    mu_global = float(np.mean(rq_color_bar))
    Gamma_bg = math.cos(math.pi * abs(mu_global - 0.5))
    mask_w = (prob_w > Config.MASK_THRESH).astype(np.uint8)
    mask_b = (prob_b > Config.MASK_THRESH).astype(np.uint8)
    c_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features_list = []
    valid_cw, valid_cb = [], []
    sum_Ek = 0.0
    for is_white, contours in [(True, c_w), (False, c_b)]:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3.0:
                continue
            M = cv2.moments(cnt)
            cx = M['m10'] / M['m00'] if M['m00'] != 0 else cnt[0][0][0]
            cy = M['m01'] / M['m00'] if M['m00'] != 0 else cnt[0][0][1]
            pts = cnt.squeeze(1)
            if pts.ndim == 1:
                pts = np.expand_dims(pts, axis=0)
            distances = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            ratio = np.min(distances) / max(np.max(distances), 1e-6)
            mask = np.zeros_like(rq_color_bar, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            mu_k = cv2.mean(rq_color_bar, mask=mask)[0]
            color_penalty = abs(2 * mu_k - 1.0)
            Ek = (ratio ** Config.W5) * math.exp(-Config.W3 * color_penalty)
            sum_Ek += Ek
            features_list.append({"area": area, "ratio": ratio, "mu_k": mu_k, "Ek": Ek})
            if is_white:
                valid_cw.append(cnt)
            else:
                valid_cb.append(cnt)
    N = len(features_list)
    P6 = 2.0 * abs(mu_global - 0.5)
    if N == 0:
        score_asdi_raw = Config.W6 * P6
        score_asdi_bad = max(0.0, min(score_asdi_raw * 100.0, 100.0))
        score_asdi = 100.0 - score_asdi_bad
        score_mmcso = 10.0 * math.log10(1.0 + (Config.W3 + Config.W5 + Config.W6*(1-P6)) / 1e-6)
        score_eedcf = (Gamma_bg ** Config.W6) * 100.0
        return score_eedcf, score_asdi, score_mmcso, 0, 0, valid_cw, valid_cb
    A_def = sum([f["area"] for f in features_list])
    A_avg = A_def / N
    P1 = min(A_def / Config.TOL_AREA, 1.0)
    P2 = min(A_avg / Config.TOL_AVG_AREA, 1.0)
    P4 = min(N / Config.TOL_COUNT, 1.0)
    P3 = sum([1.0 - abs(2 * f["mu_k"] - 1.0) for f in features_list]) / N
    P5 = sum([1.0 - f["ratio"] for f in features_list]) / N
    score_asdi_raw = Config.W1*P1 + Config.W2*P2 + Config.W3*P3 + Config.W4*P4 + Config.W5*P5 + Config.W6*P6
    score_asdi_bad = max(0.0, min(score_asdi_raw * 100.0, 100.0))
    score_asdi = 100.0 - score_asdi_bad
    Signal = Config.W3*(1-P3) + Config.W5*(1-P5) + Config.W6*(1-P6)
    Noise = Config.W1*P1 + Config.W2*P2 + Config.W4*P4
    score_mmcso = 10.0 * math.log10(1.0 + Signal / (Noise + 1e-6))
    H_defect = math.sqrt((Config.W1 * P1)**2 + (Config.W2 * P2)**2 + (Config.W4 * P4)**2)
    mean_Ek = sum_Ek / N
    score_eedcf_raw = (Gamma_bg ** Config.W6) * (mean_Ek / (1.0 + H_defect))
    score_eedcf = max(0.0, min(score_eedcf_raw * 100.0, 100.0))
    return score_eedcf, score_asdi, score_mmcso, len(valid_cw), len(valid_cb), valid_cw, valid_cb

print("Initializing multi-dimensional surface quality detection model...")
global_model = AFMBWNetV2_Full().to(Config.DEVICE)
if os.path.exists(Config.MODEL_PATH):
    global_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    global_model.eval()
    print("Model loaded successfully!")
else:
    print(f"Warning: Model weight file not found {Config.MODEL_PATH}")

def setup_record_directory():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S_Batch")
    batch_dir = os.path.join(Config.RECORD_DIR, date_str, time_str)
    orig_dir = os.path.join(batch_dir, "Originals")
    pred_dir = os.path.join(batch_dir, "Predictions")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return batch_dir, orig_dir, pred_dir

def process_batch(file_objs, progress=gr.Progress()):
    if not file_objs:
        return [], pd.DataFrame(), "Please upload images first"
    gallery_images, results_data = [], []
    for idx, file_obj in progress.tqdm(enumerate(file_objs), desc="Performing high-precision analysis...", total=len(file_objs)):
        file_path = file_obj.name
        original_filename = os.path.basename(file_path)
        timestamp = int(time.time() * 1000)
        img_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        z_raw = img.astype(np.float32) if img.ndim == 2 else (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.float32)
        z_raw_resized = cv2.resize(z_raw, (Config.TARGET_SIZE, Config.TARGET_SIZE))
        z01_for_net = robust_norm01(z_raw_resized.copy())
        z01_c = clahe01(detrend_by_gaussian(z01_for_net))
        ch2, ch3 = local_rms(z01_c), grad_mag(z01_c)
        x = torch.from_numpy(np.stack([z01_c, ch2, ch3])).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(global_model(x)[0][0]).cpu().numpy()
        s_eedcf, s_asdi, s_mmcso, nw, nb, cw, cb = calculate_multidimensional_metrics(prob[0], prob[1], z_raw_resized)
        canvas = cv2.cvtColor(np.clip(robust_norm01(z_raw_resized) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.drawContours(canvas, cw, -1, Config.COLOR_WHITE, Config.THICKNESS)
        cv2.drawContours(canvas, cb, -1, Config.COLOR_BLACK, Config.THICKNESS)
        caption = f"EEDCF: {s_eedcf:.1f} | ASDI: {s_asdi:.1f} | MMCSO: {s_mmcso:.1f}dB"
        gallery_images.append((canvas, caption))
        results_data.append({
            "Filename": original_filename,
            "Detection Code": str(timestamp),
            "EEDCF (Energy Entropy Comprehensive Representation) ▲": f"{s_eedcf:.2f}",
            "ASDI (Surface Deterioration Index) ▲": f"{s_asdi:.2f}",
            "MMCSO (Morphology Signal-to-Noise Ratio) ▲": f"{s_mmcso:.1f} dB",
            "White Protrusions": nw,
            "Black Pits": nb,
            "Ranking Score": float(s_eedcf)
        })
    df = pd.DataFrame(results_data)
    if not df.empty:
        df = df.sort_values(by="Ranking Score", ascending=False).drop(columns=["Ranking Score"])
        msg = f"Analysis completed! Processed {len(file_objs)} images."
    else:
        msg = "Failed to process any images."
    return gallery_images, df, msg

custom_css = ""

with gr.Blocks(title="AFM Multi-dimensional Surface Quality Intelligent Analysis Terminal", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML()
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("# AFM Surface Quality Analysis")
            upload_files = gr.File(label="Drag or click to upload AFM images (.tif / .jpg)", file_count="multiple", file_types=["image"])
            process_btn = gr.Button("Start Multi-dimensional Joint Evaluation", variant="primary", size="lg")
            gr.HTML()
            gr.Markdown("### Quality Metrics Explanation")
            gr.Markdown("- **EEDCF (Energy Entropy Comprehensive Representation)**: Range 0-100, higher values indicate better surface quality")
            gr.Markdown("- **ASDI (Surface Deterioration Index)**: Range 0-100, higher values indicate better surface quality")
            gr.Markdown("- **MMCSO (Morphology Signal-to-Noise Ratio)**: Higher dB values indicate better morphology quality")
            status_text = gr.Textbox(label="System Run Logs", interactive=False, lines=2)
        with gr.Column(scale=5):
            gr.Markdown("# Visualization Results")
            output_gallery = gr.Gallery(label="Visualization Results", show_label=False, columns=2, height=550, object_fit="contain")
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Data Report")
            output_df = gr.Dataframe(label="Data Report (Sorted by EEDCF Quality in Descending Order)", interactive=False, wrap=True)
    process_btn.click(
        fn=process_batch,
        inputs=upload_files,
        outputs=[output_gallery, output_df, status_text]
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False)
