"""Strip the JEOL info bar (black strip with magnification/scale-bar text) burned
into the bottom of every exported SEM image, so it never contaminates the
measurement region."""
import numpy as np


def detect_bar_top(gray, dark_thresh=60, row_dark_frac=0.5):
    """Scan up from the bottom row; a row belongs to the info bar if most of its
    pixels are near-black. Returns the row index where the real image ends."""
    h = gray.shape[0]
    bar_top = h
    for y in range(h - 1, -1, -1):
        row = gray[y]
        dark_frac = float(np.mean(row < dark_thresh))
        if dark_frac > row_dark_frac:
            bar_top = y
        else:
            break
    # sanity fallback: JEOL exports are typically 1024 (960 image + 64 bar) or a
    # uniform resize thereof (bar = 6.25% of total height). If detection collapsed
    # to nothing or ate an implausibly large chunk, fall back to that ratio.
    frac = (h - bar_top) / h
    if frac < 0.02 or frac > 0.15:
        bar_top = int(round(h * (1 - 64 / 1024)))
    return bar_top


def crop_bar(gray):
    top = detect_bar_top(gray)
    return gray[:top, :], top
