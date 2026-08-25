"""Core boundary-detection algorithm.

SEM cross-section close-up of a solder/IMC/Cu interface (BSE COMPO contrast),
built with zero labeled ground truth. Solder alloy here is SnBi or SnBiAgCu.

Physical model, confirmed against a domain expert's direct labeling of
several images (not just inferred from Z-contrast theory):
    Cu electrode  -- darkest real phase, forms the bottom of the frame.
    IMC (Cu6Sn5)  -- single uniform-grayscale phase, lighter than Cu. Its
                      Cu-facing boundary is nearly flat/horizontal; its
                      solder-facing boundary is scalloped (a diffusion
                      growth front, so irregular by nature).
    Solder        -- multi-phase (Sn/Bi/Ag sub-phases, visibly different
                      gray levels), heterogeneous/granular in appearance.
                      Over-etching preferentially attacks Sn, so heavily
                      over-etched solder can look pitted and near-black from
                      topography, not low-Z signal -- that's an etching
                      artifact, not a real dark phase, and must fold into
                      solder, not get mistaken for Cu.
Both Cu and IMC are hard phases and keep visible polishing scratch marks;
solder is soft and smears during polishing (hence its granular look).
Scratches are therefore NOT a Cu-specific signature -- they mark "Cu or
IMC", not "Cu alone".

Region-identification is judged purely on whether it matches this physical
picture (shape, uniformity, position) -- NOT on whether the resulting
thickness number looks like a smooth trend across aging time/alloy. A nice
monotonic-looking trend is not evidence the segmentation is correct, and
chasing one is a good way to quietly bias the region judgment.

Approach:
1. Cu detection, cheap-first: the darkest Otsu band, bottom-connected, is
   Cu on most alloys/etching conditions and hugs the true (jagged) Cu
   surface more precisely than a smoothed seam. Only when that fails --
   confirmed on the SnBi-100 condition, where the solder matrix can be
   etched dark enough to out-dark Cu itself -- fall back to a texture seam
   (Cu's smooth texture doesn't depend on its absolute brightness).
2. Multi-level Otsu (2 thresholds) splits the whole image into dark/mid/
   bright bands. Which band is IMC is NOT fixed across alloys (confirmed:
   brightest band on one alloy, mid-brightness on another) -- so for each
   band, take the sub-components that actually touch the Cu mask (Cu
   excluded first, since Cu itself came from one of these same bands and
   would otherwise trivially "win"), and pick whichever band forms the
   widest continuous touching band. That's IMC; the rest above Cu is
   solder (this also folds isolated over-etched-Sn islands that don't
   touch Cu into solder).
3. Thickness = count of IMC-classified pixels per column (area method),
   which stays correct for scalloped/interdigitated IMC shapes. Kept
   downstream of the region judgment, not feeding back into it.

Every threshold is computed from each image's own histogram -- nothing is
fit to a labeled thickness value.
"""
import numpy as np
import cv2
from scipy.signal import medfilt, find_peaks


def otsu_threshold(values, bins=256):
    """Standard 1-threshold Otsu over an arbitrary 1-D sample."""
    hist, _ = np.histogram(values, bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return bins // 2
    idx = np.arange(bins)
    csum = np.cumsum(hist)
    cmean = np.cumsum(hist * idx)
    global_mean = cmean[-1] / total
    best_var, best_t = -1.0, bins // 2
    for t in range(1, bins - 1):
        w0 = csum[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        m0 = cmean[t] / w0
        m1 = (cmean[-1] - cmean[t]) / w1
        var_between = w0 * (m0 - global_mean) ** 2 + w1 * (m1 - global_mean) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def is_bimodal(values, min_peak_frac=0.02, min_valley_drop=0.25, min_separation=10):
    """Rough bimodality test on a smoothed histogram: two real humps with a
    real valley between them, not just Otsu splitting one hump in half."""
    if values.size < 200:
        return False
    hist, _ = np.histogram(values, bins=128, range=(0, 256))
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return False
    hist_smooth = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peaks, props = find_peaks(hist_smooth, height=hist_smooth.max() * min_peak_frac)
    if len(peaks) < 2:
        return False
    order = np.argsort(props["peak_heights"])[::-1]
    p1, p2 = sorted(peaks[order[:2]])
    if (p2 - p1) * 2 < min_separation:
        return False
    valley = hist_smooth[p1:p2 + 1].min()
    lower_peak = min(hist_smooth[p1], hist_smooth[p2])
    if lower_peak == 0:
        return False
    if valley / lower_peak > (1 - min_valley_drop):
        return False
    return True


def local_std_map(gray, k=9):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sq_mean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sq_mean - mean * mean, 0, None)
    return np.sqrt(var)


def multi_otsu_2thresh(values, bins=256):
    """2-threshold Otsu: split a 1-D intensity sample into 3 classes by
    maximizing between-class variance. Brute-force over the histogram (256
    bins -> ~32k threshold pairs), which is cheap and exact."""
    hist, _ = np.histogram(values, bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return bins // 3, 2 * bins // 3
    idx = np.arange(bins)
    csum = np.cumsum(hist)
    cmean = np.cumsum(hist * idx)
    global_mean = cmean[-1] / total
    best_var, best_t1, best_t2 = -1.0, bins // 3, 2 * bins // 3
    for t1 in range(1, bins - 2):
        w0 = csum[t1]
        if w0 == 0:
            continue
        m0 = cmean[t1] / w0
        for t2 in range(t1 + 1, bins - 1):
            w1 = csum[t2] - csum[t1]
            if w1 == 0:
                continue
            w2 = total - csum[t2]
            if w2 == 0:
                break
            m1 = (cmean[t2] - cmean[t1]) / w1
            m2 = (cmean[-1] - cmean[t2]) / w2
            var_between = (w0 * (m0 - global_mean) ** 2 + w1 * (m1 - global_mean) ** 2
                           + w2 * (m2 - global_mean) ** 2)
            if var_between > best_var:
                best_var, best_t1, best_t2 = var_between, t1, t2
    return best_t1, best_t2


def dp_ridge_path(strength, max_jump, smooth_lambda):
    """Trace the y(x) seam across all columns maximizing cumulative `strength`
    minus a quadratic penalty for vertical jumps between adjacent columns.
    Rows pre-set to -inf in `strength` are excluded (restricts the search
    band). Returns (path, per-column achieved score)."""
    h, w = strength.shape
    dp = np.full((h, w), -np.inf, dtype=np.float64)
    back = np.zeros((h, w), dtype=np.int32)
    dp[:, 0] = strength[:, 0]

    offsets = np.arange(-max_jump, max_jump + 1)
    penalties = smooth_lambda * (offsets.astype(np.float64) ** 2)
    rows = np.arange(h)
    for x in range(1, w):
        prev = dp[:, x - 1]
        best_val = np.full(h, -np.inf)
        best_off = np.zeros(h, dtype=np.int32)
        for dy, pen in zip(offsets, penalties):
            src = rows - dy
            valid = (src >= 0) & (src < h)
            cand = np.full(h, -np.inf)
            cand[valid] = prev[src[valid]] - pen
            better = cand > best_val
            best_val = np.where(better, cand, best_val)
            best_off = np.where(better, dy, best_off)
        dp[:, x] = strength[:, x] + best_val
        back[:, x] = best_off

    if not np.any(np.isfinite(dp[:, -1])):
        return None, None

    path = np.zeros(w, dtype=np.int32)
    path[-1] = int(np.argmax(dp[:, -1]))
    for x in range(w - 1, 0, -1):
        path[x - 1] = path[x] - back[path[x], x]
    path = np.clip(path, 0, h - 1)
    return path, strength[path, np.arange(w)]


SEED_SEARCH_BAND_FRAC = 0.35
SEED_MAX_JUMP = 4
SEED_SMOOTH_LAMBDA = 3.0
BACKGROUND_PERCENTILE = 40   # texture level considered "smooth" (Cu)
RUN_LEN = 6                  # consecutive smooth rows required to confirm the Cu edge
MAX_EXPAND_FRAC = 0.35
MEDIAN_KERNEL = 21
SMOOTH_KERNEL = 7


def _smooth(a, mk=MEDIAN_KERNEL, bk=SMOOTH_KERNEL):
    a = medfilt(a, kernel_size=mk)
    kernel = np.ones(bk) / bk
    pad = bk // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, kernel, mode="valid")


def _find_cu_boundary(gray_dn, std_map):
    h, w = gray_dn.shape
    band_top = int(SEED_SEARCH_BAND_FRAC * h)
    std_band = std_map.copy()
    std_band[:band_top, :] = -1e9
    seed, seed_scores = dp_ridge_path(std_band, SEED_MAX_JUMP, SEED_SMOOTH_LAMBDA)
    if seed is None:
        return None, "IMC 텍스처 능선을 찾지 못함."

    typical = float(np.median(std_map))
    seam_level = float(np.median(seed_scores))
    if seam_level <= typical * 1.5:
        return None, (f"텍스처 능선 대비가 배경 수준과 큰 차이가 없음 "
                       f"(seam {seam_level:.1f} vs background {typical:.1f}).")

    bg_level = float(np.percentile(std_map, BACKGROUND_PERCENTILE))
    max_extent = int(MAX_EXPAND_FRAC * h)
    cu_top = np.full(w, np.nan)
    for x in range(w):
        col = std_map[:, x]
        y = y0 = int(seed[x])
        limit = min(h - 1, y0 + max_extent)
        while y < limit:
            if np.all(col[y:min(h, y + RUN_LEN)] < bg_level):
                break
            y += 1
        cu_top[x] = y
    return _smooth(cu_top), None


MIN_CU_WIDTH_COVERAGE = 0.6


def _keep_components_touching_row(mask_u8, row_index):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    touching = set(np.unique(labels[row_index, :])) - {0}
    return np.isin(labels, list(touching)) if touching else np.zeros_like(mask_u8, dtype=bool)


def _find_cu_mask(gray, gray_dn, std_map, t1):
    """Cu detection, cheap-first: the darkest Otsu band, bottom-connected, is
    Cu on most alloys/etching conditions and hugs the true (jagged) Cu
    surface more precisely than a smoothed seam. Only when that fails --
    confirmed on the SnBi-100 condition, where the solder matrix is etched
    dark enough to out-dark Cu itself -- fall back to the texture-seam method
    (alloy-agnostic: Cu's smooth texture, not its absolute brightness)."""
    h, w = gray.shape
    dark = gray_dn <= t1
    dark_open = cv2.morphologyEx((dark * 255).astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    cu_mask = _keep_components_touching_row(dark_open, h - 1)
    coverage = float(np.mean(cu_mask.any(axis=0)))
    if coverage >= MIN_CU_WIDTH_COVERAGE:
        return cu_mask, "dark-otsu-bottom-connected", None

    cu_top, err = _find_cu_boundary(gray_dn, std_map)
    if cu_top is None:
        return None, None, err
    cu_mask = np.zeros((h, w), dtype=bool)
    for x in range(w):
        cu_mask[int(cu_top[x]):, x] = True
    return cu_mask, "texture-seam-fallback", None


# Cu/IMC boundary is nearly flat/horizontal (confirmed by the domain expert:
# unrelated to scratches, unrelated to the scalloped IMC/solder edge) --
# enforce that with heavy smoothing rather than tracing local brightness
# noise pixel-by-pixel.
INNER_MEDIAN_KERNEL = 61
INNER_SMOOTH_KERNEL = 21


def _refine_cu_mask(gray_dn, cu_mask):
    """cu_mask (from _find_cu_mask) is Cu+IMC combined whenever the two sit
    close enough in brightness that the outer dark/mid/bright split couldn't
    tell them apart on its own (confirmed on SnBi-100: IMC there is bright
    enough, relative to Cu, to blend into what the outer split calls 'Cu').
    Peel IMC back out: only if cu_mask's own brightness is genuinely bimodal
    (two real phases in there, not one uniform Cu), split it with a LOCAL
    Otsu (computed on cu_mask's pixels only, so solder's brightness can't
    drag the threshold) and keep just the darker sub-population as Cu -- as
    a heavily-smoothed, near-flat line per column, since a jagged result here
    would just be chasing brightness noise inside what should be one phase."""
    h, w = gray_dn.shape
    vals = gray_dn[cu_mask]
    if not is_bimodal(vals):
        return cu_mask, np.zeros((h, w), dtype=bool)

    t_inner = otsu_threshold(vals)
    cu_sub = cu_mask & (gray_dn <= t_inner)

    split_row = np.full(w, np.nan)
    for x in range(w):
        ys = np.where(cu_mask[:, x])[0]
        if not ys.size:
            continue
        cu_ys = np.where(cu_sub[:, x])[0]
        split_row[x] = cu_ys.min() if cu_ys.size else ys.max() + 1

    valid_cols = ~np.isnan(split_row)
    if not valid_cols.any():
        return cu_mask, np.zeros((h, w), dtype=bool)
    idx = np.arange(w)
    filled = np.interp(idx, idx[valid_cols], split_row[valid_cols])
    split_smooth = _smooth(filled, INNER_MEDIAN_KERNEL, INNER_SMOOTH_KERNEL)

    new_cu = np.zeros((h, w), dtype=bool)
    peeled = np.zeros((h, w), dtype=bool)
    for x in range(w):
        if not cu_mask[:, x].any():
            continue
        split = int(round(split_smooth[x]))
        new_cu[split:, x] = cu_mask[split:, x]
        peeled[:split, x] = cu_mask[:split, x]
    return new_cu, peeled


# IMC's solder-facing edge is a real scallop (a diffusion growth front), but
# the raw touching-component mask picks up thin stray tendrils/speckle where
# it happens to connect through a narrow gap into solder's own texture --
# noise, not additional IMC. Smooth the upper envelope moderately (enough to
# drop spikes/speckle, not so much it flattens the real scallop curve) and
# clip everything above it away, per the domain expert's description.
ENVELOPE_MEDIAN_KERNEL = 17
ENVELOPE_SMOOTH_KERNEL = 9


def _smooth_imc_envelope(imc_mask, cu_mask):
    h, w = imc_mask.shape
    top = np.full(w, np.nan)
    bottom = np.full(w, np.nan)  # = Cu top for this column (bottom of IMC)
    for x in range(w):
        imc_ys = np.where(imc_mask[:, x])[0]
        cu_ys = np.where(cu_mask[:, x])[0]
        if imc_ys.size:
            top[x] = imc_ys.min()
        if cu_ys.size:
            bottom[x] = cu_ys.min()

    valid = ~np.isnan(top) & ~np.isnan(bottom)
    if not valid.any():
        return imc_mask
    idx = np.arange(w)
    top_filled = np.interp(idx, idx[valid], top[valid])
    bottom_filled = np.interp(idx, idx[valid], bottom[valid])
    top_smooth = _smooth(top_filled, ENVELOPE_MEDIAN_KERNEL, ENVELOPE_SMOOTH_KERNEL)

    new_mask = np.zeros((h, w), dtype=bool)
    rows = np.arange(h)[:, None]
    lo = np.clip(top_smooth, 0, h)[None, :]
    hi = np.clip(bottom_filled, 0, h)[None, :]
    band = (rows >= lo) & (rows < hi)
    new_mask[:, valid] = band[:, valid]
    return new_mask


def segment_interface(gray):
    h, w = gray.shape
    gray_dn = cv2.medianBlur(gray, 3)
    k = max(7, (min(h, w) // 100) | 1)
    std_map = local_std_map(gray_dn, k=k)

    t1, t2 = multi_otsu_2thresh(gray_dn.ravel())
    cu_mask, cu_method, err = _find_cu_mask(gray, gray_dn, std_map, t1)
    peeled_imc = None
    if cu_mask is not None:
        cu_mask, peeled_imc = _refine_cu_mask(gray_dn, cu_mask)

    result = {
        "width": w, "height": h, "applicable": cu_mask is not None,
        "cu_mask": cu_mask, "imc_mask": None, "solder_mask": None,
        "cu_method": cu_method, "confidence_note": err or "",
    }
    if cu_mask is None:
        return result
    classes = {
        "dark": gray_dn <= t1,
        "mid": (gray_dn > t1) & (gray_dn <= t2),
        "bright": gray_dn > t2,
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cu_dilated = cv2.dilate((cu_mask * 255).astype(np.uint8), kernel, iterations=2)

    # Which brightness band is IMC is NOT fixed across alloys/etching
    # conditions (confirmed by inspection: IMC is the brightest band on one
    # alloy, mid-brightness on another). So don't hardcode it -- for each of
    # the 3 Otsu bands, take the sub-components that actually touch the
    # (reliable) Cu boundary, and pick whichever band forms the widest
    # continuous touching band. The other two bands are solder (this also
    # correctly folds isolated over-etched-Sn dark islands that don't touch
    # Cu into solder).
    best_name, best_mask, best_cov = None, None, -1.0
    for name, cls_mask in classes.items():
        # exclude Cu itself first -- Cu was carved out of one of these same
        # Otsu bands, so without this a band containing Cu trivially "touches"
        # cu_dilated everywhere and always wins by default
        cls_u8 = ((cls_mask & ~cu_mask) * 255).astype(np.uint8)
        n, labels, _, _ = cv2.connectedComponentsWithStats(cls_u8, connectivity=8)
        touch_labels = set(np.unique(labels[(cu_dilated > 0) & (cls_u8 > 0)])) - {0}
        touching = np.isin(labels, list(touch_labels)) if touch_labels else np.zeros((h, w), dtype=bool)
        cov = float(np.mean(touching.any(axis=0)))
        if cov > best_cov:
            best_name, best_mask, best_cov = name, touching, cov

    imc_u8 = cv2.morphologyEx((best_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    imc_mask = (imc_u8 > 0) & ~cu_mask
    imc_mask = _smooth_imc_envelope(imc_mask, cu_mask)
    solder_mask = ~cu_mask & ~imc_mask

    result["cu_mask"] = cu_mask
    result["imc_mask"] = imc_mask
    result["solder_mask"] = solder_mask
    result["thresholds"] = (t1, t2)
    result["imc_class"] = best_name
    result["imc_touch_coverage"] = best_cov
    return result


def thickness_from_segmentation(seg):
    w = seg["width"]

    if not seg["applicable"]:
        return {
            "status": "not_applicable",
            "reason": "Cu 하단부와 텍스처 대비가 뚜렷한 경계를 찾지 못함. "
                      "계면 근접 촬영 이미지가 아니거나 배율/구도가 다를 수 있음 — 수동 확인 필요. "
                      + seg.get("confidence_note", ""),
        }

    imc_mask = seg["imc_mask"]
    thickness_px = imc_mask.sum(axis=0).astype(float)
    valid = thickness_px > 0

    n_valid = int(np.sum(valid))
    pct_valid = n_valid / w if w else 0.0
    if n_valid == 0:
        return {"status": "failed",
                "reason": "IMC로 분류된 픽셀이 Cu 경계와 맞닿은 열이 없음 — 자동 측정 불가, 수동 확인 필요."}

    vals = thickness_px[valid]
    result = {
        "status": "ok", "mode": "texture-seam Cu boundary + grayscale phase grouping (area method)",
        "solder_detected": True,
        "thickness_px": thickness_px, "valid_mask": valid,
        "mean_px": float(np.mean(vals)), "median_px": float(np.median(vals)),
        "std_px": float(np.std(vals)), "min_px": float(np.min(vals)), "max_px": float(np.max(vals)),
        "n_columns": w, "n_valid": n_valid, "pct_valid": pct_valid,
    }

    imc_top = np.full(w, np.nan)
    for x in range(w):
        ys = np.where(imc_mask[:, x])[0]
        if ys.size:
            imc_top[x] = ys.min()
    diffs = np.abs(np.diff(imc_top[valid]))
    jaggedness = float(np.mean(diffs)) if diffs.size else 0.0
    if pct_valid >= 0.85 and jaggedness <= 3.0:
        result["confidence"] = "medium-high (자동 검출 — 오버레이 육안 확인 권장)"
    elif pct_valid >= 0.5:
        result["confidence"] = "medium (일부 열 미검출/불안정 — 오버레이 확인 필요)"
    else:
        result["confidence"] = "low (다수 열에서 IMC-Cu 접촉 불안정 — 수동 확인 필요)"

    return result
