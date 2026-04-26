# utils.py
from . import config

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalize_thai_digits(text):
    if text is None:
        return ""
    table = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
    return str(text).translate(table)


def clean_digits(text):
    text = normalize_thai_digits(text)
    return re.sub(r"[^0-9]", "", str(text))


def resolve_image_path(item):
    rel = str(item.get("image_rel") or "").replace("\\", "/")

    # 1) path ตรงจาก manifest
    if rel:
        p = config.BASE_DIR / rel
        if p.exists():
            return p

    # 2) ชื่อไฟล์ตรง
    filename = Path(rel).name
    if filename:
        p = config.SCORE_CROPS_DIR / filename
        if p.exists():
            return p

    # 3) fallback: record_id + row_index
    record_id = item.get("record_id")
    row_index = item.get("row_index")

    if record_id is not None and row_index is not None:
        prefix = f"{record_id}_row_{int(row_index):02d}_"
        matches = sorted(config.SCORE_CROPS_DIR.glob(prefix + "*.png"))
        if matches:
            return matches[0]

    return None


def prepare_image(img):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    scale = 1.0

    if h < config.MIN_CROP_HEIGHT:
        scale = max(scale, config.MIN_CROP_HEIGHT / h)

    if w < config.MIN_CROP_WIDTH:
        scale = max(scale, config.MIN_CROP_WIDTH / w)

    scale = min(scale, 3.5)

    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # sharpen เบา ๆ ให้เลขจางชัดขึ้น
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.45, blur, -0.45, 0)

    img = cv2.copyMakeBorder(
        img,
        top=24,
        bottom=24,
        left=24,
        right=24,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return img


def crop_digit_area(img):
    """
    crop เฉพาะเลขคะแนนฝั่งซ้ายสุด ก่อนวงเล็บ
    ไม่เอาคำอ่านไทยฝั่งขวา
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]

    x1 = 0
    x2 = int(w * config.DIGIT_CROP_RATIO)

    y1 = 0
    y2 = h

    crop = img[y1:y2, x1:x2].copy()
    return prepare_image(crop)


def save_csv(rows, path):
    tmp = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")

    try:
        tmp.replace(path)
    except PermissionError:
        backup = path.with_name(path.stem + "_backup.csv")
        tmp.replace(backup)
        print(f"WARNING: {path.name} locked. Saved backup to {backup.name}")


def make_key(item):
    return f"{item.get('record_id')}__{item.get('row_index')}"


def load_checkpoint(path):
    if not path.exists():
        return {}, []

    df = pd.read_csv(path)
    done = {}

    for _, row in df.iterrows():
        key = f"{row.get('record_id')}__{row.get('row_index')}"
        done[key] = row.to_dict()

    return done, df.to_dict("records")

def crop_digit_area_verify(img):
    """
    crop ซ้ายกว้างขึ้นเพื่ออ่านยืนยันอีกครั้ง
    ใช้กันเคส Qwen อ่าน crop แรกผิดแต่ดัน accept
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]

    x1 = 0
    x2 = int(w * config.VERIFY_DIGIT_CROP_RATIO)

    crop = img[0:h, x1:x2].copy()
    return prepare_image(crop)

def crop_digit_area_wide(img):
    """
    crop ซ้ายกว้างกว่า verify เพื่อใช้เป็นรอบตัดสิน majority
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]

    x1 = 0
    x2 = int(w * config.WIDE_DIGIT_CROP_RATIO)

    crop = img[0:h, x1:x2].copy()
    return prepare_image(crop)

def crop_word_area(img):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]

    x1 = int(w * config.WORD_CROP_X1_RATIO)
    x2 = int(w * config.WORD_CROP_X2_RATIO)

    crop = img[0:h, x1:x2].copy()
    return prepare_image(crop)

def handwriting_score(img):
    if img is None or img.size == 0:
        return 0, 0, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # เฉพาะหมึกเข้มจริง ลดการนับพื้น/เส้นจาง
    mask = (gray < 130).astype("uint8") * 255

    dark_ratio = float(mask.mean() / 255.0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    count = 0
    total_area = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < 16:
            continue

        # ignore dotted line / tiny noise
        if h <= 7 and w <= 14:
            continue

        # ignore long table line
        if w > 150 and h < 14:
            continue

        # handwriting-like component
        if area >= 22 and h >= 8:
            count += 1
            total_area += int(area)

    return count, total_area, dark_ratio


def is_blank_cell(digit_img, word_img):
    digit_count, digit_area, digit_ratio = handwriting_score(digit_img)
    word_count, word_area, word_ratio = handwriting_score(word_img)

    total_count = digit_count + word_count
    total_area = digit_area + word_area
    total_ratio = digit_ratio + word_ratio

    # blank จริงต้องไม่มีทั้ง component และ dark ratio ต่ำ
    return (
        total_count < config.MIN_HANDWRITING_COMPONENTS
        and total_area < config.MIN_HANDWRITING_AREA
        and total_ratio < config.BLANK_DARK_RATIO
    )

def detect_zero_like_digit(img):
    """
    ใช้ image processing ช่วยจับเลข 0:
    - มี contour คล้ายวง/oval
    - ไม่ใช่เส้นเดี่ยวแบบเลข 1
    - ไม่ใช่เส้นหักแบบ 7
    """
    if img is None or img.size == 0:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold หมึกเข้ม
    _, th = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    # ลบ noise เล็ก ๆ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < config.ZERO_LOOP_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        aspect = w / float(h)
        extent = area / float(w * h)

        # เลข 0 มักเป็นวงรี/วงปิด: กว้างและสูงพอ, aspect ไม่ผอมมาก
        if 0.35 <= aspect <= 1.45 and extent >= config.ZERO_LOOP_MIN_RATIO and h >= 18 and w >= 10:
            return True

    return False


def crop_digit_area_tight(img):
    """
    crop แคบกว่า digit crop ปกติ เอาไว้ช่วยเช็ค 0 เฉพาะบริเวณเลขจริง
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    x2 = int(w * 0.25)
    crop = img[0:h, 0:x2].copy()
    return prepare_image(crop)