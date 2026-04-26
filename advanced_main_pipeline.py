from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import re

import cv2
import fitz
import numpy as np

from src.digit_classifier import DigitCascadeRecognizer, DigitClassifierConfig, resolve_cnn_path
from src.onnx_ocr import OnnxOCRConfig, OnnxThaiOCR, resolve_onnx_paths
from src.pipeline_types import PipelineResult, RowExtraction
from src.yolo_layout import LayoutConfig, YoloLayoutDetector, find_model_path


class AdvancedOCRPipeline:
    def __init__(self, models_dir: str = "models"):
        self.project_dir = Path(__file__).resolve().parent
        self.models_dir = Path(models_dir)
        self.candidate_master = load_constituency_master(self.project_dir / "candidate_master_constituency.json")
        self.party_master = load_party_master(self.project_dir / "party_master_party_list.json")
        self.layout = YoloLayoutDetector(
            LayoutConfig(model_path=find_model_path(self.models_dir, "doclayout_yolov10n.pt"))
        )
        det_path, rec_path, rec_config_path = resolve_onnx_paths(self.models_dir)
        self.printed_ocr = OnnxThaiOCR(
            OnnxOCRConfig(
                det_model_path=det_path,
                rec_model_path=rec_path,
                rec_config_path=rec_config_path,
            )
        )
        self.digit_recognizer = DigitCascadeRecognizer(
            DigitClassifierConfig(
                cnn_model_path=resolve_cnn_path(self.models_dir),
            )
        )

    def pdf_to_image(self, pdf_path: str | Path, page_index: int = 0, dpi: int = 200) -> np.ndarray:
        doc = fitz.open(str(pdf_path))
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def image_to_rows(self, image: np.ndarray) -> PipelineResult:
        boxes = self.layout.detect(image, page_index=1)
        paired_rows = self.layout.pair_rows(boxes, image.shape)
        warnings: list[str] = []
        if not paired_rows:
            warnings.append("No layout boxes detected")

        rows: list[RowExtraction] = []
        for row_index, (name_box, score_box) in enumerate(paired_rows):
            name_crop = self.layout.crop_box(image, name_box)
            score_crop = self.layout.crop_box(image, score_box)

            name_result = self.printed_ocr.recognize_crop(name_crop)
            digit_result = self.digit_recognizer.recognize(score_crop)

            review_required = bool(
                digit_result.review_required
                or name_box is None
                or score_box is None
                or not name_result.text
            )
            row = RowExtraction(
                row_index=row_index,
                name_box=name_box,
                score_box=score_box,
                name=name_result.text,
                name_confidence=name_result.confidence,
                score=digit_result.text,
                score_confidence=digit_result.confidence,
                score_source=digit_result.source,
                review_required=review_required,
                metadata={
                    "name_ocr_source": name_result.source,
                    "digit_candidates": digit_result.candidates,
                },
            )
            rows.append(row)

        status = "ok"
        if warnings or any(row.review_required for row in rows):
            status = "review_required"

        return PipelineResult(
            file="",
            page=1,
            status=status,
            rows=rows,
            warnings=warnings,
            metadata={
                "detected_boxes": len(boxes),
                "detector_classes": sorted({box.label for box in boxes}),
            },
        )

    def run(self, pdf_path: str | Path) -> PipelineResult:
        image = self.pdf_to_image(pdf_path)
        result = self.image_to_rows(image)
        result.file = str(pdf_path)
        result.metadata["input_shape"] = list(image.shape)
        try:
            result.metadata["ocr_detector_probe"] = self.printed_ocr.validate_detector(image)
        except Exception as exc:
            result.warnings.append(f"OCR detector probe failed: {exc}")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced federated local OCR pipeline")
    parser.add_argument("input_path", help="PDF or image path to process")
    parser.add_argument("--models-dir", default="models", help="Directory containing local model weights")
    parser.add_argument("--output-json", default="", help="Optional JSON output path")
    parser.add_argument(
        "--output-format",
        choices=("intermediate", "raw"),
        default="intermediate",
        help="Output a model-handoff friendly schema or raw row extraction",
    )
    return parser.parse_args()


def load_image_or_pdf(input_path: str, pipeline: AdvancedOCRPipeline) -> PipelineResult:
    suffix = Path(input_path).suffix.lower()
    if suffix == ".pdf":
        return pipeline.run(input_path)
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read input: {input_path}")
    result = pipeline.image_to_rows(image)
    result.file = input_path
    result.metadata["input_shape"] = list(image.shape)
    return result


def main() -> None:
    args = parse_args()
    pipeline = AdvancedOCRPipeline(models_dir=args.models_dir)
    result = load_image_or_pdf(args.input_path, pipeline)
    payload: dict[str, Any]
    if args.output_format == "raw":
        payload = result.to_dict()
    else:
        payload = build_intermediate_payload(result, pipeline.candidate_master, pipeline.party_master)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")

def build_intermediate_payload(
    result: PipelineResult,
    candidate_master: dict[int, dict[str, Any]] | None = None,
    party_master: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_path = Path(result.file)
    form_type = detect_form_type(result.file)
    records: list[dict[str, Any]] = []
    any_review = result.status != "ok"

    for row in result.rows:
        text = normalize_text(row.name)
        entry_no = row.row_index + 1
        score_value = parse_score(row.score, row.score_confidence, row.review_required)

        candidate_review = bool(
            row.review_required or score_value is None
        )
        any_review = any_review or candidate_review

        if form_type == "party_list":
            master_entry = (party_master or {}).get(entry_no, {})
            records.append(
                {
                    "row_index": row.row_index,
                    "party_no": entry_no,
                    "party_no_confidence": 1.0,
                    "party_name": master_entry.get("party_name"),
                    "party_name_confidence": 1.0 if master_entry.get("party_name") else 0.0,
                    "score": score_value,
                    "score_confidence": row.score_confidence if score_value is not None else 0.0,
                    "score_source": row.score_source,
                    "review_required": candidate_review,
                    "raw_party_text": text or None,
                    "raw_score_text": row.score or None,
                }
            )
        else:
            master_entry = (candidate_master or {}).get(entry_no, {})
            records.append(
                {
                    "row_index": row.row_index,
                    "candidate_no": entry_no,
                    "candidate_no_confidence": 1.0,
                    "candidate_name": master_entry.get("candidate_name"),
                    "candidate_name_confidence": 1.0 if master_entry.get("candidate_name") else 0.0,
                    "party_name": master_entry.get("party_name"),
                    "party_name_confidence": 1.0 if master_entry.get("party_name") else 0.0,
                    "score": score_value,
                    "score_confidence": row.score_confidence if score_value is not None else 0.0,
                    "score_source": row.score_source,
                    "review_required": candidate_review,
                    "raw_name_text": text or None,
                    "raw_score_text": row.score or None,
                }
            )

    payload = {
        "schema_version": "intermediate_v1",
        "source_file": result.file,
        "source_basename": source_path.name,
        "source_parent": str(source_path.parent),
        "page": result.page,
        "form_type": form_type,
        "total_valid_votes": None,
        "total_valid_votes_confidence": 0.0,
        "review_required": any_review,
        "num_rows": len(records),
        "warnings": result.warnings,
        "metadata": {
            **result.metadata,
            "raw_rows": [row.to_dict() for row in result.rows],
        },
    }
    if form_type == "party_list":
        payload["party_master_applied"] = bool(party_master)
        payload["party_results"] = records
    else:
        payload["candidate_master_applied"] = bool(candidate_master)
        payload["candidates"] = records
    return payload


def normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("\u200b", "")
    return cleaned


def split_candidate_and_party(text: str) -> tuple[str | None, str | None]:
    if not text:
        return (None, None)

    if any(title in text for title in ("นาย", "นาง", "น.ส.", "นางสาว", "ดร.")):
        # If a person title exists, keep the full text as candidate_name for now.
        return (text, None)

    # Current OCR mostly yields party-like text, so treat it as party_name.
    return (None, text)


def parse_score(text: str, confidence: float, review_required: bool) -> int | None:
    if not text or review_required or confidence <= 0:
        return None
    if not text.isdigit():
        return None
    if len(text) > 3:
        return None
    if len(text) > 1 and text.startswith("0"):
        return None
    return int(text)


def load_constituency_master(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    return {
        int(item["candidate_no"]): {
            "candidate_name": item.get("candidate_name"),
            "party_name": item.get("party_name"),
        }
        for item in candidates
    }


def load_party_master(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    parties = payload.get("parties", [])
    return {
        int(item["party_no"]): {
            "party_name": item.get("party_name"),
        }
        for item in parties
    }


def detect_form_type(source_file: str) -> str:
    normalized = re.sub(r"[\s._\-()\\/\[\]{}]+", "", str(source_file))
    if "\u0e1a\u0e0a" in normalized:
        return "party_list"
    return "constituency"


if __name__ == "__main__":
    main()
