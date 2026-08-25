import cv2
import numpy as np


def draw_overlay(gray, seg, result, px_size_um=None):
    """Colour-fill the three classified phases so a person can see exactly
    which pixels were counted as IMC (orange) vs Cu (green) vs solder (blue,
    includes over-etched-Sn artifact pixels) before trusting the number."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if result["status"] != "ok":
        cv2.putText(vis, "MEASUREMENT FAILED - manual check needed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, result.get("reason", "")[:90], (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return vis

    fill = vis.copy()
    if seg.get("solder_mask") is not None:
        fill[seg["solder_mask"]] = (255, 80, 0)   # blue = solder (incl. reclassified artifact)
    fill[seg["cu_mask"]] = (0, 200, 0)            # green = Cu
    fill[seg["imc_mask"]] = (0, 140, 255)          # orange = IMC
    vis = cv2.addWeighted(vis, 0.55, fill, 0.45, 0)

    invalid = ~result["valid_mask"]
    if invalid.any():
        for x in np.where(invalid)[0]:
            cv2.line(vis, (int(x), 0), (int(x), h - 1), (255, 0, 255), 1, cv2.LINE_AA)

    lines = []
    if px_size_um:
        lines.append(f"mean IMC thickness: {result['mean_px'] * px_size_um:.3f} um "
                      f"(median {result['median_px'] * px_size_um:.3f}, "
                      f"std {result['std_px'] * px_size_um:.3f})")
    else:
        lines.append(f"mean thickness: {result['mean_px']:.1f} px (NO SCALE CALIBRATION)")
    lines.append(f"mode: {result['mode']}  |  confidence: {result['confidence']}")
    lines.append(f"valid columns: {result['n_valid']}/{result['n_columns']} ({result['pct_valid']:.0%})")

    band_h = 24 + len(lines) * 26
    shade = vis[0:band_h, :].copy()
    cv2.rectangle(vis, (0, 0), (w, band_h), (0, 0, 0), -1)
    vis[0:band_h, :] = cv2.addWeighted(vis[0:band_h, :], 0.6, shade, 0.4, 0)

    y0 = 22
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = h - 10
    legend = "green=Cu  orange=IMC  blue=solder(+etch artifact)  magenta col=flagged"
    cv2.putText(vis, legend, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, legend, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return vis
