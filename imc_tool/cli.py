import argparse
import sys
from pathlib import Path

import pandas as pd

from . import metadata, barcrop, segment, visualize, naming
from .io_utils import imread_unicode, imwrite_unicode

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process_one(image_path, txt_index=None, manual_scale_um_per_px=None, meta_search_dirs=()):
    image_path = Path(image_path)
    gray_full = imread_unicode(image_path)
    gray, bar_top = barcrop.crop_bar(gray_full)

    meta = None
    px_size_um = manual_scale_um_per_px
    scale_source = "manual" if manual_scale_um_per_px else None
    if px_size_um is None:
        if txt_index is not None:
            meta = metadata.find_metadata_indexed(image_path, txt_index)
        else:
            meta = metadata.find_metadata_for_image(image_path, extra_search_dirs=meta_search_dirs)
        if meta:
            px_size_um = metadata.pixel_size_um(meta, gray_full.shape[1])
            scale_source = meta["source"]

    seg = segment.segment_interface(gray)
    result = segment.thickness_from_segmentation(seg)

    cond = naming.parse_condition(image_path.stem)

    row = {
        "file": str(image_path),
        "alloy": cond["alloy"],
        "aging_hours": cond["aging_hours"],
        "idx": cond["idx"],
        "status": result["status"],
        "scale_source": scale_source,
        "px_size_um": px_size_um,
    }

    if result["status"] == "ok":
        if px_size_um:
            row["mean_thickness_um"] = result["mean_px"] * px_size_um
            row["median_thickness_um"] = result["median_px"] * px_size_um
            row["std_thickness_um"] = result["std_px"] * px_size_um
            row["min_thickness_um"] = result["min_px"] * px_size_um
            row["max_thickness_um"] = result["max_px"] * px_size_um
        row["mean_thickness_px"] = result["mean_px"]
        row["mode"] = result["mode"]
        row["solder_detected"] = result["solder_detected"]
        row["confidence"] = result["confidence"]
        row["pct_valid_columns"] = result["pct_valid"]
        row["needs_review"] = (not px_size_um) or ("low" in result["confidence"])
    else:
        row["needs_review"] = True
        row["reason"] = result.get("reason", "")

    return row, gray, seg, result, px_size_um


def cmd_single(args):
    meta_dirs = [args.meta_root] if args.meta_root else []
    row, gray, seg, result, px_size_um = process_one(
        args.image, manual_scale_um_per_px=args.scale_um_per_px, meta_search_dirs=meta_dirs)
    for k, v in row.items():
        print(f"{k}: {v}")

    if args.overlay:
        vis = visualize.draw_overlay(gray, seg, result, px_size_um)
        imwrite_unicode(args.overlay, vis)
        print(f"overlay saved: {args.overlay}")


def cmd_batch(args):
    root = Path(args.folder)
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"found {len(images)} images under {root}")

    txt_index = metadata.build_txt_index(args.meta_root) if args.meta_root else metadata.build_txt_index(root)

    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, img_path in enumerate(images, 1):
        try:
            row, gray, seg, result, px_size_um = process_one(img_path, txt_index=txt_index)
        except Exception as e:
            rows.append({"file": str(img_path), "status": "error", "reason": str(e), "needs_review": True})
            print(f"[{i}/{len(images)}] ERROR {img_path.name}: {e}")
            continue

        rows.append(row)
        flag = "REVIEW" if row.get("needs_review") else "ok"
        thick = row.get("mean_thickness_um")
        thick_s = f"{thick:.3f} um" if thick is not None else "n/a"
        print(f"[{i}/{len(images)}] {img_path.name}: {row['status']} {thick_s} [{flag}]")

        if overlay_dir:
            vis = visualize.draw_overlay(gray, seg, result, px_size_um)
            out_name = img_path.stem + "_overlay.png"
            imwrite_unicode(overlay_dir / out_name, vis)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nsaved {len(df)} rows to {args.out}")
    if "needs_review" in df.columns:
        n_review = int(df["needs_review"].sum())
        print(f"flagged for manual review: {n_review}/{len(df)} ({n_review / len(df):.0%})")


def main():
    parser = argparse.ArgumentParser(description="SEM cross-section IMC layer thickness tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="process one image and print/optionally save an overlay")
    p_single.add_argument("image")
    p_single.add_argument("--overlay", help="path to save annotated overlay PNG")
    p_single.add_argument("--scale-um-per-px", type=float, default=None,
                           help="manual override if no JEOL .txt sidecar is found")
    p_single.add_argument("--meta-root", default=None,
                           help="root to search for a same-named JEOL .txt sidecar if none sits next to the image")
    p_single.set_defaults(func=cmd_single)

    p_batch = sub.add_parser("batch", help="process every image under a folder, write a summary CSV")
    p_batch.add_argument("folder")
    p_batch.add_argument("--out", default="imc_thickness_results.csv")
    p_batch.add_argument("--overlay-dir", default=None, help="if set, save an overlay PNG per image here")
    p_batch.add_argument("--meta-root", default=None,
                          help="root to search for JEOL .txt sidecars if images don't carry their own "
                               "(defaults to the input folder itself)")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
