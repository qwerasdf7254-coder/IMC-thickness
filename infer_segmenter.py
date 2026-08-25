"""Apply the trained SmallUNet to a SEM image (or a whole folder) and produce
the same kind of thickness measurement + overlay as imc_tool.cli, but driven
by the learned model's Cu/IMC/solder classification instead of the rule-based
heuristics in imc_tool/segment.py.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

from train_segmenter import SmallUNet, CKPT_PATH, DEVICE
import sys
sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from imc_tool.io_utils import imread_unicode, imwrite_unicode
from imc_tool.barcrop import crop_bar
from imc_tool import metadata


def load_model():
    model = SmallUNet(n_classes=3).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_mask(model, gray, tile=256, overlap=64):
    """Sliding-window inference over the full image (model was trained on
    256x256 crops), averaging logits in overlap regions for smoother seams."""
    h, w = gray.shape
    logits_sum = np.zeros((3, h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                patch = gray[y0:y1, x0:x1]
                ph, pw = patch.shape
                pad_h, pad_w = tile - ph, tile - pw
                patch_p = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                t = torch.from_numpy(patch_p.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
                out = model(t)[0].cpu().numpy()  # (3, tile, tile)
                logits_sum[:, y0:y1, x0:x1] += out[:, :ph, :pw]
                weight[y0:y1, x0:x1] += 1

    logits_sum /= np.maximum(weight, 1)
    pred = np.argmax(logits_sum, axis=0).astype(np.uint8)  # 0=solder,1=IMC,2=Cu
    return pred


def largest_component_per_class(pred):
    """Light cleanup: keep only the largest connected component per class to
    drop small speckle mis-classifications, without altering true shape."""
    out = pred.copy()
    for cls in (1, 2):
        mask = (pred == cls).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        drop = mask.astype(bool) & (labels != keep)
        out[drop] = 0  # reassign speckle back to solder(background) by default
    return out


from scipy.signal import medfilt


def _smooth_line(arr, w, median_k, box_k):
    mk = min(median_k, w - (1 - w % 2))
    mk = mk - 1 if mk % 2 == 0 else mk
    mk = max(1, mk)
    s = medfilt(arr, kernel_size=mk)
    bk = max(1, min(box_k, w))
    kernel = np.ones(bk) / bk
    return np.convolve(np.pad(s, bk // 2, mode="edge"), kernel, mode="valid")[:w]


def _despike(arr, w, window_frac=0.35, mad_k=5.0, min_window=31):
    """Reject columns whose raw boundary position deviates far from a wide
    local-median reference, then re-interpolate over them. A polishing
    scratch (a strong, spatially-extended texture shared by Cu and IMC) can
    drag the raw per-column boundary along its own diagonal for many
    adjacent columns -- that run is long enough that a same-sized median
    filter treats it as signal, not noise, and just follows it. Comparing
    against a much wider reference window catches the drag regardless of
    its run length, as long as the scratch doesn't span more than about
    half that window."""
    window = max(min_window, int(round(w * window_frac)))
    window = window - 1 if window % 2 == 0 else window
    window = min(window, w - (1 - w % 2))
    window = max(1, window)
    ref = medfilt(arr, kernel_size=window)
    dev = np.abs(arr - ref)
    mad = np.median(dev)
    thresh = mad_k * max(mad, 1.0)
    bad = dev > thresh
    if not bad.any() or bad.all():
        return arr
    idx = np.arange(w)
    good = ~bad
    return np.interp(idx, idx[good], arr[good])


def smooth_imc_solder_boundary(pred, cu_median_k=61, cu_box_k=21,
                                imc_median_k_frac=0.55, imc_median_k_range=(9, 51),
                                imc_box_k_frac=0.35, imc_box_k_range=(5, 25)):
    """The labels this model was trained on are smooth hand-drawn curves;
    pixel-level jaggedness in the prediction is model noise, not signal.

    Cu/IMC (lower) boundary: physically near-flat (confirmed against a domain
    expert's labeling -- unrelated to scratches or the scalloped IMC/solder
    edge), so smooth it with a wide fixed kernel and then hard-assign every
    pixel below it to Cu. Noise inside solid Cu (e.g. a polishing scratch the
    model mis-reads as IMC/solder) must not survive once we know it's below
    the interface -- there is nothing else it could physically be.

    IMC/solder (upper) boundary: a genuine scallop, so the smoothing window
    must scale with the local IMC thickness. A fixed-size kernel flattens
    thin IMC layers (where the kernel spans most of the whole layer) far more
    than thick ones, which is exactly backwards -- thin layers need a
    *narrower* kernel to keep their real undulation visible."""
    h, w = pred.shape
    top = np.full(w, np.nan)
    bottom = np.full(w, np.nan)
    for x in range(w):
        imc_ys = np.where(pred[:, x] == 1)[0]
        cu_ys = np.where(pred[:, x] == 2)[0]
        if imc_ys.size:
            top[x] = imc_ys.min()
        if cu_ys.size:
            bottom[x] = cu_ys.min()
        elif imc_ys.size:
            bottom[x] = imc_ys.max() + 1

    valid = ~np.isnan(top) & ~np.isnan(bottom)
    if not valid.any():
        return pred
    idx = np.arange(w)
    top_f = np.interp(idx, idx[valid], top[valid])
    bottom_f = np.interp(idx, idx[valid], bottom[valid])

    top_f = _despike(top_f, w)
    bottom_f = _despike(bottom_f, w)

    bottom_s = _smooth_line(bottom_f, w, cu_median_k, cu_box_k)

    thickness = np.clip(bottom_f[valid] - top_f[valid], 1, None)
    typical_thickness = float(np.median(thickness))
    median_k = int(round(typical_thickness * imc_median_k_frac))
    median_k = max(imc_median_k_range[0], min(imc_median_k_range[1], median_k))
    if median_k % 2 == 0:
        median_k += 1
    box_k = int(round(typical_thickness * imc_box_k_frac))
    box_k = max(imc_box_k_range[0], min(imc_box_k_range[1], box_k))
    top_s = _smooth_line(top_f, w, median_k, box_k)

    out = pred.copy()
    rows = np.arange(h)[:, None]
    lo = np.clip(top_s, 0, h)[None, :]
    hi = np.clip(bottom_s, 0, h)[None, :]
    new_imc = (rows >= lo) & (rows < hi)
    new_cu = rows >= hi
    out[:, valid] = np.where(new_imc[:, valid], 1, np.where(new_cu[:, valid], 2, 0))
    return out


def process(model, image_path, meta_search_dirs=()):
    image_path = Path(image_path)
    gray_full = imread_unicode(image_path)
    gray, bar_top = crop_bar(gray_full)

    pred = predict_mask(model, gray)
    pred = largest_component_per_class(pred)
    pred = smooth_imc_solder_boundary(pred)

    imc_mask = pred == 1
    cu_mask = pred == 2
    thickness_px = imc_mask.sum(axis=0).astype(float)
    valid = thickness_px > 0

    meta = metadata.find_metadata_for_image(image_path, extra_search_dirs=meta_search_dirs)
    px_size_um = metadata.pixel_size_um(meta, gray_full.shape[1]) if meta else None

    return {
        "file": str(image_path), "pred": pred, "gray": gray,
        "imc_mask": imc_mask, "cu_mask": cu_mask,
        "thickness_px": thickness_px, "valid": valid,
        "px_size_um": px_size_um,
        "mean_px": float(thickness_px[valid].mean()) if valid.any() else float("nan"),
        "pct_valid": float(valid.mean()),
    }


def draw_overlay(gray, pred, result):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fill = vis.copy()
    fill[pred == 2] = (0, 200, 0)
    fill[pred == 1] = (0, 140, 255)
    vis = cv2.addWeighted(vis, 0.55, fill, 0.45, 0)
    px = result["px_size_um"]
    if px:
        text = f"mean IMC thickness: {result['mean_px']*px:.3f} um  (model, valid {result['pct_valid']:.0%})"
    else:
        text = f"mean thickness: {result['mean_px']:.1f} px (no scale)  valid {result['pct_valid']:.0%}"
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="image file or folder")
    ap.add_argument("--meta-root", default=r"E:\00_정명진\03_연구실 컴퓨터\03_SBSAC_Reliability")
    ap.add_argument("--overlay-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = load_model()
    target = Path(args.target)
    files = [target] if target.is_file() else sorted(target.rglob("*.jpg"))

    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, f in enumerate(files, 1):
        try:
            r = process(model, f, meta_search_dirs=[args.meta_root])
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {f.name}: {e}")
            continue
        um = r["mean_px"] * r["px_size_um"] if r["px_size_um"] else float("nan")
        print(f"[{i}/{len(files)}] {f.name}: {um:.3f} um  valid={r['pct_valid']:.0%}")
        rows.append((f.name, um, r["pct_valid"]))
        if overlay_dir:
            vis = draw_overlay(r["gray"], r["pred"], r)
            imwrite_unicode(overlay_dir / f"{f.stem}_model_overlay.png", vis)

    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            w.writerow(["file", "mean_thickness_um", "pct_valid"])
            w.writerows(rows)
        print("saved", args.out)
