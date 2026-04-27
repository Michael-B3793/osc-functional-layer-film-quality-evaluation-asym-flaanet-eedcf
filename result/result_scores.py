from pathlib import Path
import csv
import sys
import io
import warnings
from contextlib import redirect_stdout
import numpy as np
import cv2
import torch


def compute_scores(image_path: Path, app):
    img_data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        z_raw = img.astype(np.float32)
    else:
        z_raw = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.float32)
    z_raw_resized = cv2.resize(z_raw, (app.Config.TARGET_SIZE, app.Config.TARGET_SIZE))
    z01_for_net = app.robust_norm01(z_raw_resized.copy())
    z01_c = app.clahe01(app.detrend_by_gaussian(z01_for_net))
    ch2 = app.local_rms(z01_c)
    ch3 = app.grad_mag(z01_c)
    x = torch.from_numpy(np.stack([z01_c, ch2, ch3])).unsqueeze(0).to(app.Config.DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(app.global_model(x)[0][0]).cpu().numpy()
    s_eedcf, s_asdi, s_mmcso, _, _, _, _ = app.calculate_multidimensional_metrics(prob[0], prob[1], z_raw_resized)
    return float(s_eedcf), float(s_asdi), float(s_mmcso)


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    warnings.filterwarnings("ignore")
    with redirect_stdout(io.StringIO()):
        import app
    data_csv = root / "result" / "data.csv"
    image_dir = root / "result" / "data_images"
    rows = []
    with data_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    csv_map = {}
    for r in rows:
        name = r["Filename"].strip()
        csv_map[name] = {
            "EEDCF": float(r["EEDCF"]),
            "ASDI": float(r["ASDI"]),
            "MMCSO_dB": float(r["MMCSO_dB"]),
        }
    print("Filename,EEDCF,ASDI,MMCSO_dB")
    for name in sorted(csv_map.keys(), key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x):
        p = image_dir / name
        if not p.exists():
            continue
        scores = compute_scores(p, app)
        if scores is None:
            continue
        eedcf, asdi, mmcso = scores
        print(f"{name},{eedcf:.8f},{asdi:.8f},{mmcso:.9f}")


if __name__ == "__main__":
    main()
