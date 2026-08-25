"""Per-image grayscale-brightness profile of the model's IMC region,
normalized per image so the darkest pixel maps to 1 and the brightest
pixel maps to 0 (in this BSE/Z-contrast SnBi(AgCu) system, darkest ~ Cu
[Z=29, the darkest real phase per the domain notes in segment.py's
docstring], brightest ~ the highest-Z sub-phase visible in solder, which
for Bi [Z=83] would be far brighter than Sn [Z=50] or Ag [Z=47]).

Normalizing per image on its own min/max removes session-to-session
brightness/gain differences (confirmed real and substantial across this
dataset -- see the grayscale-normalization discussion earlier in this
project) so IMC "color" can be compared across alloy/aging conditions on
a common 0..1 scale instead of raw, incomparable pixel values.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\82109\IMC_Thickness_Tool")
from infer_segmenter import load_model, process

_PATTERN = re.compile(r"^(?P<alloy>\d+)_(?P<aging>\d+)_(?:f_)?(?P<idx>\d+)$")


def parse_condition(stem):
    m = _PATTERN.match(stem)
    if not m:
        return {"alloy": None, "aging_hours": None, "idx": None}
    return {"alloy": m.group("alloy"), "aging_hours": int(m.group("aging")), "idx": int(m.group("idx"))}


def imc_normalized_stats(result):
    gray = result["gray"]
    imc_mask = result["imc_mask"]
    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax <= gmin:
        return None
    # darkest -> 1, brightest -> 0
    norm = (gmax - gray.astype(np.float64)) / (gmax - gmin)
    vals = norm[imc_mask]
    if vals.size == 0:
        return None
    return {
        "imc_norm_mean": float(vals.mean()),
        "imc_norm_median": float(np.median(vals)),
        "imc_norm_std": float(vals.std()),
        "imc_norm_p10": float(np.percentile(vals, 10)),
        "imc_norm_p90": float(np.percentile(vals, 90)),
        "n_imc_px": int(vals.size),
        "gray_min": gmin,
        "gray_max": gmax,
    }


def main():
    root = Path(r"E:\00_정명진\03_연구실 컴퓨터\03_SBSAC_Reliability\IMC 측정")
    files = sorted(root.rglob("*.jpg"))
    print(f"found {len(files)} images under {root}")

    model = load_model()
    rows = []
    for i, f in enumerate(files, 1):
        cond = parse_condition(f.stem)
        try:
            r = process(model, f)
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {f.name}: {e}")
            continue
        stats = imc_normalized_stats(r)
        if stats is None:
            print(f"[{i}/{len(files)}] {f.name}: no IMC pixels / degenerate brightness range")
            continue
        row = {"file": f.name, **cond, **stats}
        rows.append(row)
        print(f"[{i}/{len(files)}] {f.name}: mean={stats['imc_norm_mean']:.3f} "
              f"median={stats['imc_norm_median']:.3f} std={stats['imc_norm_std']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("imc_grayscale_normalized.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved imc_grayscale_normalized.csv ({len(df)} rows)")

    # "IMC 측정" has a handful of files present under both its top-level and
    # main/ subfolder (rglob walks both) -- dedup by filename before
    # aggregating so a duplicated image isn't silently double-weighted in a
    # per-condition mean built from only 4-5 images.
    n_dupes = int(df["file"].duplicated().sum())
    if n_dupes:
        print(f"dropping {n_dupes} duplicate file(s) before per-condition aggregation")
    df_unique = df.drop_duplicates(subset="file", keep="first")

    agg = (df_unique.groupby(["alloy", "aging_hours"])["imc_norm_mean"]
             .agg(["mean", "std", "count"])
             .reset_index()
             .sort_values(["alloy", "aging_hours"]))
    agg.to_csv("imc_grayscale_by_condition.csv", index=False, encoding="utf-8-sig")
    print(f"saved imc_grayscale_by_condition.csv ({len(agg)} condition groups)")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
