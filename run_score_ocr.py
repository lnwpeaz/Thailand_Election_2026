#run_score_ocr.py
import argparse

import pandas as pd

from score_ocr import config, client, utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--debug-map", action="store_true")
    return parser.parse_args()


def is_repeated_noise(text):
    text = utils.clean_digits(text)
    return len(text) >= 5 and len(set(text)) <= 2


def is_valid_score(text):
    text = utils.clean_digits(text)

    if not text:
        return False

    if len(text) > config.MAX_DIGITS:
        return False

    if len(text) > 1 and text.startswith("0"):
        return False

    if is_repeated_noise(text):
        return False

    return True


def extract_raw_part(qwen_raw, key):
    """
    ดึงค่า digit=... หรือ word=... จาก qwen_raw_text
    เช่น digit=17 | word=1
    """
    import re

    text = str(qwen_raw)
    m = re.search(rf"{key}=([0-9]+)", text)
    if m:
        return m.group(1)
    return ""


def choose_score(qwen_digits, qwen_raw, ocr_mode):
    qwen_digits = utils.clean_digits(qwen_digits)
    qwen_raw = str(qwen_raw)

    if ocr_mode == "blank_cell":
        return None, "blank", False, "blank_cell"

    if "ERROR:" in qwen_raw:
        return None, "review_required", True, "qwen_error"

    if qwen_raw.strip().upper() == "UNKNOWN":
        return None, "review_required", True, "qwen_unknown"

    if not is_valid_score(qwen_digits):
        return None, "review_required", True, "no_valid"

    # เคสมั่นใจอยู่แล้ว
    if ocr_mode in [
        "digit_primary_fast",
        "digit_word_agree",
        "digit_only_ambiguous_accept",
        "word_fallback_accept",
        "zero_local_accept",
    ]:
        return int(qwen_digits), "qwen", False, ocr_mode

    # เคสเลขซ้ายกับคำอ่านไม่ตรง
    if ocr_mode == "digit_word_mismatch":
        digit_part = extract_raw_part(qwen_raw, "digit")
        word_part = extract_raw_part(qwen_raw, "word")

        # 1) ถ้าเลขยาว 3 หลักขึ้นไป มักเป็น hallucination เช่น 217, 372, 454
        if len(qwen_digits) >= 3:
            return int(qwen_digits), "qwen_unverified", True, "mismatch_long_number"

        # 2) ถ้า word อ่านเป็นเลขยาวแปลก ๆ เช่น 100, 109 ให้ review
        if len(word_part) >= 3:
            return int(qwen_digits), "qwen_unverified", True, "mismatch_word_long"

        # 3) เคส 7 vs 1/5/6 เสี่ยงมาก ให้ review
        if qwen_digits == "7" and word_part in {"1", "5", "6"}:
            return int(qwen_digits), "qwen_unverified", True, "mismatch_7_risky"

        # 4) ที่เหลือ accept แบบ soft
        return int(qwen_digits), "qwen_soft_accept", False, "mismatch_soft_accept"

    # fallback ถ้าได้เลข valid ก็ accept
    return int(qwen_digits), "qwen_soft_accept", False, "accepted_default"


def base_row(item):
    return {
        "id": f"{item.get('record_id')}_row_{int(item.get('row_index')):02d}",
        "record_id": item.get("record_id"),
        "batch_index": item.get("batch_index"),
        "row_index": item.get("row_index"),
        "form_type": item.get("form_type"),
        "source_parent": item.get("source_parent"),
        "source_basename": item.get("source_basename"),
        "image_rel": item.get("image_rel"),
        "raw_score_text": item.get("raw_score_text"),
    }


def error_row(item, reason):
    row = base_row(item)
    row.update({
        "qwen_raw_text": "",
        "ocr_text": "",
        "ocr_mode": "",
        "final_score": None,
        "final_source": reason,
        "need_review": True,
        "review_reason": reason,
        "manual_score": "",
        "review_note": "",
    })
    return row


def main():
    args = parse_args()

    if args.reset and config.CHECKPOINT_CSV.exists():
        config.CHECKPOINT_CSV.unlink()
        print("Deleted checkpoint")

    manifest = utils.load_json(config.MANIFEST_PATH)

    if args.limit > 0:
        manifest = manifest[:args.limit]

    done_map, rows = utils.load_checkpoint(config.CHECKPOINT_CSV)

    for i, item in enumerate(manifest, start=1):
        key = utils.make_key(item)

        if key in done_map:
            continue

        img_path = utils.resolve_image_path(item)

        if args.debug_map and i <= 20:
            print(
                "MAP:",
                item.get("record_id"),
                item.get("row_index"),
                "=>",
                img_path.name if img_path else None,
            )

        if img_path is None:
            rows.append(error_row(item, "missing_image"))
            continue

        img = utils.imread_unicode(img_path)

        if img is None:
            rows.append(error_row(item, "unreadable_image"))
            continue

        qwen_digits, qwen_raw, ocr_mode = client.predict_score_multi(img)

        final_score, final_source, need_review, review_reason = choose_score(
            qwen_digits,
            qwen_raw,
            ocr_mode,
        )

        row = base_row(item)
        row.update({
            "qwen_raw_text": qwen_raw,
            "ocr_text": qwen_digits,
            "ocr_mode": ocr_mode,
            "final_score": final_score,
            "final_source": final_source,
            "need_review": need_review,
            "review_reason": review_reason,
            "manual_score": "",
            "review_note": "",
        })

        rows.append(row)

        if len(rows) % 10 == 0:
            print(f"{len(rows)}/{len(manifest)}")

        if len(rows) % config.SAVE_EVERY == 0:
            utils.save_csv(rows, config.CHECKPOINT_CSV)
            print("checkpoint saved")

    utils.save_csv(rows, config.CHECKPOINT_CSV)
    utils.save_csv(rows, config.OUTPUT_CSV)

    df = pd.DataFrame(rows)

    print("DONE")
    print("total:", len(df))
    print("accepted:", int((df["need_review"] == False).sum()))
    print("review:", int((df["need_review"] == True).sum()))
    print("missing_image:", int((df["final_source"] == "missing_image").sum()))
    print("saved:", config.OUTPUT_CSV)


if __name__ == "__main__":
    main()