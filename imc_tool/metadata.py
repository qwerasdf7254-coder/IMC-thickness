"""Read JEOL SEM sidecar .txt metadata to get an exact px-to-um scale for an image,
and fall back to searching sibling capture folders for a same-named sidecar when the
image itself has none (e.g. images copied into a curated folder without their .txt)."""
import re
from pathlib import Path

_MARKER_RE = re.compile(r"([0-9.]+)\s*([a-zA-Zµμ]+)")


def parse_jeol_txt(txt_path):
    """Return dict with mag, micron_bar_px, micron_marker_um, full_size(w,h), title, date/time.
    Returns None if the file doesn't look like a JEOL SEM_DATA sidecar."""
    data = {}
    try:
        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if "SEM_DATA_VERSION" not in text:
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        # lines are stored as "<idx>\t<$KEY> <value...>" when read via cat -n style tools,
        # but the raw file is just "$KEY value...". Strip a leading index+tab if present.
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].isdigit():
            line = parts[1]
        if line.startswith("$$SM_MICRON_BAR"):
            data["micron_bar_px"] = float(line.split()[1])
        elif line.startswith("$$SM_MICRON_MARKER"):
            m = _MARKER_RE.match(line.split(None, 1)[1].strip())
            if m:
                value, unit = m.groups()
                value = float(value)
                unit = unit.lower()
                if unit in ("nm",):
                    value /= 1000.0
                elif unit in ("mm",):
                    value *= 1000.0
                # um / μm / µm all treated as micrometers
                data["micron_marker_um"] = value
        elif line.startswith("$CM_MAG"):
            data["mag"] = float(line.split()[1])
        elif line.startswith("$CM_FULL_SIZE"):
            w, h = line.split()[1:3]
            data["full_size"] = (int(float(w)), int(float(h)))
        elif line.startswith("$CM_TITLE"):
            data["title"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
        elif line.startswith("$CM_DATE"):
            data["date"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
        elif line.startswith("$CM_SIGNAL "):
            data["signal"] = line.split()[1]

    if "micron_bar_px" not in data or "micron_marker_um" not in data:
        return None
    return data


def find_metadata_for_image(image_path, extra_search_dirs=()):
    """Look for a sidecar .txt next to the image; if absent (e.g. the image was
    copied into a curated folder without it), search extra_search_dirs recursively
    for a .txt with the same basename."""
    image_path = Path(image_path)
    sidecar = image_path.with_suffix(".txt")
    if sidecar.exists():
        meta = parse_jeol_txt(sidecar)
        if meta:
            meta["source"] = str(sidecar)
            return meta

    basename_txt = image_path.stem + ".txt"
    for d in extra_search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for cand in d.rglob(basename_txt):
            meta = parse_jeol_txt(cand)
            if meta:
                meta["source"] = str(cand)
                return meta
    return None


def build_txt_index(root):
    """Pre-scan a directory tree once and index every .txt sidecar by basename,
    so batch runs over ~1000 images don't each do their own rglob."""
    index = {}
    root = Path(root)
    for p in root.rglob("*.txt"):
        index.setdefault(p.stem, []).append(p)
    return index


def find_metadata_indexed(image_path, txt_index):
    image_path = Path(image_path)
    sidecar = image_path.with_suffix(".txt")
    if sidecar.exists():
        meta = parse_jeol_txt(sidecar)
        if meta:
            meta["source"] = str(sidecar)
            return meta
    for cand in txt_index.get(image_path.stem, []):
        meta = parse_jeol_txt(cand)
        if meta:
            meta["source"] = str(cand)
            return meta
    return None


def pixel_size_um(meta, actual_img_w):
    """Convert the metadata's scale-bar calibration (recorded against the raw
    CM_FULL_SIZE capture) to the actual pixel size of the image file on disk,
    which JEOL's exporter often resamples to a smaller resolution."""
    full_w = meta.get("full_size", (actual_img_w, None))[0]
    scale_factor = actual_img_w / float(full_w) if full_w else 1.0
    bar_px_actual = meta["micron_bar_px"] * scale_factor
    return meta["micron_marker_um"] / bar_px_actual
