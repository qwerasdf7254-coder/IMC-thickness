"""Local web app: drag-and-drop a batch of SEM images and get IMC thickness
measurements automatically, without running the CLI by hand each time.

    streamlit run app.py

Reuses the exact same model + pipeline as infer_segmenter.py's CLI --
this is a front end for it, not a separate implementation.
"""
import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from infer_segmenter import load_model, process, draw_overlay

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

st.set_page_config(page_title="IMC 두께 자동 측정", layout="wide")
st.title("IMC 두께 자동 측정")
st.caption("SEM 단면 이미지를 한꺼번에 올리면 학습된 모델(SmallUNet)로 IMC 두께를 자동 측정합니다.")


@st.cache_resource
def get_model():
    return load_model()


with st.sidebar:
    st.header("설정")
    manual_scale = st.number_input(
        "수동 스케일 (µm/px)",
        min_value=0.0, value=0.0, step=0.0001, format="%.6f",
        help="0으로 두면 업로드한 JEOL .txt 사이드카에서 자동으로 읽습니다. "
             "사이드카가 없는 이미지만 이 값으로 대체됩니다.",
    )
    st.caption(
        "이미지와 같은 파일명의 JEOL `.txt` 사이드카를 이미지와 함께 올리면 "
        "자동으로 스케일을 인식합니다."
    )

uploaded = st.file_uploader(
    "이미지 (+ 선택: JEOL .txt 사이드카) 업로드",
    type=[e.lstrip(".") for e in IMG_EXTS] + ["txt"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("위에 이미지를 드래그하거나 클릭해서 업로드하세요. 여러 장을 한 번에 올릴 수 있습니다.")
else:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        image_paths = []
        for f in uploaded:
            dest = tmpdir / f.name
            dest.write_bytes(f.getvalue())
            if dest.suffix.lower() in IMG_EXTS:
                image_paths.append(dest)

        if not image_paths:
            st.warning("이미지 파일이 없습니다 (.txt 사이드카만 업로드된 것 같습니다).")
        else:
            model = get_model()
            rows = []
            overlays = {}
            progress = st.progress(0.0, text=f"측정 중... (0/{len(image_paths)})")

            for i, path in enumerate(image_paths):
                try:
                    r = process(model, path, meta_search_dirs=[tmpdir],
                                manual_scale_um_per_px=(manual_scale or None))
                except Exception as e:
                    rows.append({"file": path.name, "status": "error", "reason": str(e)})
                    progress.progress((i + 1) / len(image_paths),
                                       text=f"측정 중... ({i + 1}/{len(image_paths)})")
                    continue

                px = r["px_size_um"]
                mean_um = r["mean_px"] * px if px else None
                rows.append({
                    "file": path.name,
                    "status": "ok",
                    "mean_thickness_um": mean_um,
                    "mean_thickness_px": r["mean_px"],
                    "pct_valid": r["pct_valid"],
                    "scale_um_per_px": px,
                    "needs_review": px is None,
                })
                vis = draw_overlay(r["gray"], r["pred"], r)
                ok, buf = cv2.imencode(".png", vis)
                if ok:
                    overlays[path.name] = buf.tobytes()

                progress.progress((i + 1) / len(image_paths),
                                   text=f"측정 중... ({i + 1}/{len(image_paths)})")

            progress.empty()

            df = pd.DataFrame(rows)
            n_ok = int((df["status"] == "ok").sum())
            n_review = int(df.get("needs_review", pd.Series(dtype=bool)).sum())
            st.subheader(f"결과 — {n_ok}/{len(df)}장 측정 완료" +
                         (f", {n_review}장은 스케일 없음(px 단위만)" if n_review else ""))
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("CSV 다운로드", csv_bytes,
                                file_name="imc_thickness_results.csv", mime="text/csv")

            if overlays:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, data in overlays.items():
                        zf.writestr(Path(name).stem + "_overlay.png", data)
                st.download_button("오버레이 전체 다운로드 (zip)", zip_buf.getvalue(),
                                    file_name="overlays.zip", mime="application/zip")

                st.subheader("오버레이 미리보기")
                cols = st.columns(3)
                for i, (name, data) in enumerate(overlays.items()):
                    with cols[i % 3]:
                        st.image(data, caption=name, use_container_width=True)
