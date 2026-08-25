"""Unicode-safe image I/O. cv2.imread/imwrite fail silently on paths containing
non-ASCII (Korean) characters on Windows, which every path in this project has."""
import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"failed to decode image: {path}")
    return img


def imwrite_unicode(path, img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"failed to encode image for: {path}")
    buf.tofile(str(path))
