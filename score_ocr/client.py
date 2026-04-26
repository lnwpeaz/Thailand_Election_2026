import base64
import time
import cv2
import requests

from . import config
from .utils import clean_digits


DIGIT_PROMPT = (
    "Read ONLY the handwritten vote score number in this image. "
    "It may be Arabic digits or Thai digits. Convert Thai digits to Arabic digits. "
    "Important: 0 is a closed loop or oval, do NOT read it as 7. "
    "1 is a single vertical stroke. 4 is angular/open. 7 usually has a top horizontal stroke. "
    "If the image is blank, return UNKNOWN. "
    "Return ONLY Arabic digits 0-9. No explanation."
)

WORD_PROMPT = (
    "Read ONLY the Thai handwritten number word in this image and convert it to Arabic digits. "
    "Examples: ศูนย์=0, หนึ่ง=1, สอง=2, สาม=3, สี่=4, ห้า=5, หก=6, เจ็ด=7, แปด=8, เก้า=9, "
    "สิบ=10, สิบเอ็ด=11, สิบสอง=12, สิบเจ็ด=17. "
    "If blank or unreadable, return UNKNOWN. "
    "Return ONLY Arabic digits 0-9. No explanation."
)


def _call_qwen(img, prompt):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return "", "ERROR: image_encode_failed"

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    payload = {
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 8,
        },
    }

    last_error = ""

    for _ in range(config.MAX_RETRIES):
        try:
            res = requests.post(
                config.OLLAMA_URL,
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
            res.raise_for_status()

            raw = res.json().get("response", "").strip()
            digits = clean_digits(raw)

            if len(digits) > config.MAX_DIGITS:
                digits = ""

            return digits, raw

        except Exception as e:
            last_error = str(e)
            time.sleep(1.5)

    return "", f"ERROR: {last_error}"


def has_ambiguous_digit(digits):
    return any(ch in config.AMBIGUOUS_DIGITS for ch in str(digits))


def predict_score_multi(img):
    from . import utils

    digit_img = utils.crop_digit_area(img)
    word_img = utils.crop_word_area(img)

    # blank guard แบบ conservative
    if utils.is_blank_cell(digit_img, word_img):
        return "", "digit=BLANK | word=BLANK", "blank_cell"

    digit_digits, digit_raw = _call_qwen(digit_img, DIGIT_PROMPT)

    # local zero check: ถ้า Qwen อ่านเป็น 7/1/4 แต่ภาพทรง 0 ชัด ให้แก้ candidate เป็น 0
    tight_digit_img = utils.crop_digit_area_tight(img)
    zero_like = utils.detect_zero_like_digit(tight_digit_img)

    if zero_like and digit_digits in {"1", "4", "7"}:
        digit_digits = "0"
        digit_raw = f"{digit_raw} -> local_zero_fix"

    # เลขทั่วไป ไม่ใช่กลุ่มเสี่ยง → accept เร็ว
    if digit_digits and not has_ambiguous_digit(digit_digits):
        return digit_digits, f"digit={digit_raw}", "digit_primary_fast"

    # เลขเสี่ยง หรือ digit อ่านไม่ได้ → อ่านคำอ่านช่วย
    word_digits, word_raw = _call_qwen(word_img, WORD_PROMPT)
    raw_summary = f"digit={digit_raw} | word={word_raw}"

    # ถ้า local เห็น 0 และคำอ่านไม่มีหรืออ่านไม่ออก → accept เป็น 0
    if zero_like and (not word_digits or word_digits == "0"):
        return "0", raw_summary + " | local_zero_like=True", "zero_local_accept"

    # digit ไม่ได้ แต่คำอ่านได้ → accept จากคำอ่าน
    if not digit_digits and word_digits:
        return word_digits, raw_summary, "word_fallback_accept"

    if not digit_digits and not word_digits:
        return "", raw_summary, "no_read"

    # ตรงกัน → accept
    if digit_digits and word_digits and digit_digits == word_digits:
        return digit_digits, raw_summary, "digit_word_agree"

    # mismatch แต่ถ้า digit เป็น 0 จาก local_zero_fix และ word อ่านไม่ใช่ 0 → review
    if digit_digits and word_digits and digit_digits != word_digits:
        return digit_digits, raw_summary, "digit_word_mismatch"

    # digit มี แต่ word อ่านไม่ได้ → accept ไม่ต้อง strict เกิน
    if digit_digits:
        return digit_digits, raw_summary, "digit_only_ambiguous_accept"

    return "", raw_summary, "no_read"