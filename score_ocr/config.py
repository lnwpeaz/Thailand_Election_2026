from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "score_annotation_export" / "score_annotation_manifest.json"
SCORE_CROPS_DIR = BASE_DIR / "score_annotation_export" / "score_crops"

OUTPUT_CSV = BASE_DIR / "score_ocr_results.csv"
CHECKPOINT_CSV = BASE_DIR / "score_ocr_checkpoint.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"

MAX_DIGITS = 5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 100
SAVE_EVERY = 20

MIN_CROP_HEIGHT = 220
MIN_CROP_WIDTH = 520

DIGIT_CROP_RATIO = 0.40
WORD_CROP_X1_RATIO = 0.22
WORD_CROP_X2_RATIO = 0.95

AMBIGUOUS_DIGITS = {"0", "1", "4", "7"}

# blank ให้ conservative กว่าเดิม จะได้ไม่ตัดเลขจางเป็น blank ง่าย
MIN_HANDWRITING_COMPONENTS = 1
MIN_HANDWRITING_AREA = 22
BLANK_DARK_RATIO = 0.0008

# ถ้า local image เห็นทรงวง/oval ชัด ให้ช่วยแก้เป็น 0
ZERO_LOOP_MIN_AREA = 35
ZERO_LOOP_MIN_RATIO = 0.35