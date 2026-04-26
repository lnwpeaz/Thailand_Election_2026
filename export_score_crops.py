# export_score_crops_full.py

import json
import re
from pathlib import Path

import cv2
import fitz
import numpy as np


INPUT_JSON = Path("intermediate_batch_output/master_intermediate.json")
OUTPUT_DIR = Path("score_annotation_export/score_crops")
OUTPUT_MANIFEST = Path("score_annotation_export/score_annotation_manifest.json")
FAILED_PATH = Path("score_annotation_export/score_crop_failed.json")

DPI = 200


def safe_name(text, max_len=28):
    text = str(text or "")
    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^\wก-๙]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "unknown"


def pdf_to_image(pdf_path, page_index=0, dpi=200):
    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def crop_score_box(img, score_box):
    if not score_box:
        return None

    H, W = img.shape[:2]

    x1 = int(score_box["x1"])
    y1 = int(score_box["y1"])
    x2 = int(score_box["x2"])
    y2 = int(score_box["y2"])

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # กว้าง: เอาเลข + คำอ่านไทยบางส่วน/เกือบครบ
    # ไม่ตัดแค่เลข เพราะบางแถวเลขไทย/ลายมืออ่านยาก
    nx1 = x1 - int(box_w * 0.04)
    nx2 = x2 + int(box_w * 0.02)

    # สูง: ให้เห็นตั้งแต่เส้นบนถึงเส้นล่างของแถว
    # แต่ไม่ขยายเกินจนกินแถวอื่นมาก
    ny1 = y1 - int(box_h * 0.08)
    ny2 = y2 + int(box_h * 0.08)

    nx1 = max(0, nx1)
    nx2 = min(W, nx2)
    ny1 = max(0, ny1)
    ny2 = min(H, ny2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    crop = img[ny1:ny2, nx1:nx2].copy()

    # ไม่ทำ threshold/equalize เพราะทำให้ภาพแตก
    crop = cv2.resize(crop, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)

    return crop


def imwrite_safe(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def build_filename(rec, structured, row_index):
    form_type = rec.get("form_type", "unknown")
    form_short = "party" if form_type == "party_list" else "const"

    source_parent = rec.get("source_parent", "")
    parts = Path(source_parent).parts

    # เอาท้าย ๆ ของ path เช่น อำเภอ / ตำบล / หน่วย
    useful_parts = [p for p in parts if p not in ("election_data", ".", "")]
    loc_parts = useful_parts[-3:] if useful_parts else ["unknown"]

    loc = "_".join(safe_name(p, 20) for p in loc_parts)

    no = structured.get("party_no") or structured.get("candidate_no") or row_index + 1

    return (
        f"{safe_name(rec.get('record_id'))}_"
        f"{form_short}_"
        f"{loc}_"
        f"row_{row_index:02d}_"
        f"no_{int(no):02d}.png"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    manifest = []
    failed = []

    for rec_idx, rec in enumerate(records):
        source_file = Path(rec["source_file"])
        record_id = rec["record_id"]
        form_type = rec["form_type"]

        page_number = int(rec.get("page", 1))
        page_index = max(0, page_number - 1)

        try:
            page_img = pdf_to_image(source_file, page_index=page_index, dpi=DPI)
        except Exception as e:
            failed.append({
                "record_id": record_id,
                "source_file": str(source_file),
                "error": str(e),
            })
            continue

        raw_rows = rec.get("metadata", {}).get("raw_rows", [])

        if form_type == "party_list":
            structured_rows = rec.get("party_results", [])
        else:
            structured_rows = rec.get("candidates", [])

        structured_by_index = {
            int(r["row_index"]): r for r in structured_rows
        }

        for raw in raw_rows:
            row_index = int(raw["row_index"])
            structured = structured_by_index.get(row_index, {})

            score_box = raw.get("score_box")
            crop = crop_score_box(page_img, score_box)

            if crop is None or crop.size == 0:
                failed.append({
                    "record_id": record_id,
                    "row_index": row_index,
                    "reason": "empty_crop",
                })
                continue

            filename = build_filename(rec, structured, row_index)
            out_path = OUTPUT_DIR / filename

            if not imwrite_safe(out_path, crop):
                failed.append({
                    "record_id": record_id,
                    "row_index": row_index,
                    "reason": "write_failed",
                })
                continue

            manifest.append({
                "image": str(out_path),
                "image_rel": str(out_path).replace("\\", "/"),
                "record_id": record_id,
                "batch_index": rec.get("batch_index"),
                "source_file": rec.get("source_file"),
                "source_basename": rec.get("source_basename"),
                "source_parent": rec.get("source_parent"),
                "page": rec.get("page"),
                "form_type": form_type,
                "row_index": row_index,
                "candidate_no": structured.get("candidate_no"),
                "party_no": structured.get("party_no"),
                "candidate_name": structured.get("candidate_name"),
                "party_name": structured.get("party_name"),
                "score_box": score_box,
                "raw_score_text": structured.get("raw_score_text") or raw.get("score"),
                "score_source": structured.get("score_source") or raw.get("score_source"),
                "review_required": structured.get("review_required", True),
            })

        if (rec_idx + 1) % 50 == 0:
            print(f"Processed records: {rec_idx + 1}/{len(records)}")

    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    FAILED_PATH.write_text(
        json.dumps(failed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("DONE")
    print("records:", len(records))
    print("crops:", len(manifest))
    print("failed:", len(failed))
    print("manifest:", OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()