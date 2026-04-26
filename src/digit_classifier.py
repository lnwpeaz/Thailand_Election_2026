from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import onnxruntime as ort
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.pipeline_types import DigitResult


@dataclass
class DigitClassifierConfig:
    cnn_model_path: str | None = None
    cnn_confidence_threshold: float = 0.95
    trocr_model_name: str = "microsoft/trocr-base-handwritten"
    max_digits: int = 6


class DigitCascadeRecognizer:
    def __init__(self, config: DigitClassifierConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn_session = self._load_cnn(config.cnn_model_path)
        self.processor = TrOCRProcessor.from_pretrained(config.trocr_model_name)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(config.trocr_model_name).to(self.device)

    def recognize(self, image_crop: np.ndarray | None) -> DigitResult:
        if image_crop is None or image_crop.size == 0:
            return DigitResult(
                text="",
                confidence=0.0,
                source="empty",
                review_required=True,
            )

        variants = self._build_variants(image_crop)
        cnn_candidate = self._run_cnn_sequence(variants[0])
        if cnn_candidate is not None and cnn_candidate.confidence >= self.config.cnn_confidence_threshold:
            cnn_candidate.review_required = not self._passes_rules(cnn_candidate.text)
            return cnn_candidate

        trocr_candidate = self._run_trocr_variants(variants)
        if cnn_candidate is not None:
            trocr_candidate.candidates.append(cnn_candidate.to_dict())
        trocr_candidate.review_required = not self._passes_rules(trocr_candidate.text)
        return trocr_candidate

    def _load_cnn(self, cnn_model_path: str | None):
        if not cnn_model_path:
            return None
        path = Path(cnn_model_path)
        if not path.exists():
            return None
        providers = ort.get_available_providers()
        return ort.InferenceSession(str(path), providers=providers)

    def _run_cnn_sequence(self, image_crop: np.ndarray) -> DigitResult | None:
        if self.cnn_session is None:
            return None

        segments = self._segment_digits(image_crop)
        if not segments:
            return DigitResult(
                text="",
                confidence=0.0,
                source="cnn",
                review_required=True,
            )

        input_name = self.cnn_session.get_inputs()[0].name
        text = []
        probs = []
        for segment in segments:
            tensor = self._prepare_cnn_input(segment)
            output = np.asarray(self.cnn_session.run(None, {input_name: tensor})[0])
            output = output.squeeze()
            if output.ndim != 1:
                return None
            normalized = self._softmax(output)
            idx = int(normalized.argmax())
            text.append(str(idx))
            probs.append(float(normalized[idx]))

        return DigitResult(
            text="".join(text),
            confidence=float(np.mean(probs)),
            source="cnn",
            review_required=False,
            candidates=[],
        )

    def _run_trocr_variants(self, variants: list[np.ndarray]) -> DigitResult:
        best = DigitResult(
            text="",
            confidence=0.0,
            source="trocr",
            review_required=False,
            candidates=[],
        )
        for idx, variant in enumerate(variants):
            candidate = self._run_trocr_once(variant)
            candidate.candidates.append(
                {
                    "variant_index": idx,
                    "text": candidate.text,
                    "confidence": candidate.confidence,
                    "source": candidate.source,
                }
            )
            if self._candidate_rank(candidate) > self._candidate_rank(best):
                best = candidate
        return best

    def _run_trocr_once(self, image_crop: np.ndarray) -> DigitResult:
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB) if image_crop.ndim == 3 else image_crop
        pixel_values = self.processor(images=rgb, return_tensors="pt").pixel_values.to(self.device)

        allowed_token_ids = self._digit_token_ids()
        with torch.no_grad():
            generation = self.trocr_model.generate(
                pixel_values,
                max_new_tokens=self.config.max_digits,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                bad_words_ids=[[token_id] for token_id in self._disallowed_token_ids(allowed_token_ids)],
            )
        decoded = self.processor.batch_decode(generation.sequences, skip_special_tokens=True)[0]
        filtered = "".join(ch for ch in decoded if ch.isdigit())
        confidence = self._sequence_confidence(generation.scores, generation.sequences)
        if not self._passes_rules(filtered):
            confidence = 0.0
        return DigitResult(
            text=filtered,
            confidence=confidence,
            source="trocr",
            review_required=False,
            candidates=[],
        )

    def _build_variants(self, image_crop: np.ndarray) -> list[np.ndarray]:
        variants = [image_crop]
        left_focus = self._left_focus_crop(image_crop)
        if left_focus is not None:
            variants.append(left_focus)
        numeric_window = self._extract_numeric_window(image_crop)
        if numeric_window is not None:
            variants.append(numeric_window)
            enhanced = self._enhance_for_trocr(numeric_window)
            if enhanced is not None:
                variants.append(enhanced)
            tight_left = self._left_focus_crop(numeric_window)
            if tight_left is not None:
                variants.append(tight_left)
        else:
            enhanced = self._enhance_for_trocr(image_crop)
            if enhanced is not None:
                variants.append(enhanced)
        return self._dedupe_variants(variants)

    def _left_focus_crop(self, image_crop: np.ndarray) -> np.ndarray | None:
        h, w = image_crop.shape[:2]
        if h < 8 or w < 16:
            return None
        right = max(1, int(w * 0.42))
        crop = image_crop[:, :right].copy()
        return crop if crop.size else None

    def _dedupe_variants(self, variants: list[np.ndarray]) -> list[np.ndarray]:
        deduped: list[np.ndarray] = []
        seen: set[tuple[int, int, int]] = set()
        for variant in variants:
            if variant is None or variant.size == 0:
                continue
            shape_key = (variant.shape[0], variant.shape[1], int(np.mean(variant)))
            if shape_key in seen:
                continue
            seen.add(shape_key)
            deduped.append(variant)
        return deduped

    def _extract_numeric_window(self, image_crop: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if image_crop.ndim == 3 else image_crop
        h, w = gray.shape[:2]
        if h < 12 or w < 24:
            return image_crop

        left_limit = max(1, int(w * 0.45))
        roi = gray[:, :left_limit]
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        kept_boxes: list[tuple[int, int, int, int]] = []
        min_area = max(30, int(h * w * 0.002))
        min_height = max(10, int(h * 0.22))
        for idx in range(1, num_labels):
            x, y, bw, bh, area = stats[idx]
            if area < min_area:
                continue
            if bh < min_height:
                continue
            if x == 0 and bw < int(w * 0.08):
                continue
            if x + bw >= left_limit - 2 and x > int(left_limit * 0.85):
                continue
            kept_boxes.append((x, y, x + bw, y + bh))

        if not kept_boxes:
            return image_crop[:, :left_limit].copy()

        x1 = min(box[0] for box in kept_boxes)
        y1 = min(box[1] for box in kept_boxes)
        x2 = max(box[2] for box in kept_boxes)
        y2 = max(box[3] for box in kept_boxes)

        pad_x = max(8, int((x2 - x1) * 0.25))
        pad_y = max(6, int((y2 - y1) * 0.30))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(left_limit, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return image_crop[y1:y2, x1:x2].copy()

    def _enhance_for_trocr(self, image_crop: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if image_crop.ndim == 3 else image_crop
        if gray.size == 0:
            return None
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _sequence_confidence(self, scores: tuple[torch.Tensor, ...], sequences: torch.Tensor) -> float:
        if not scores:
            return 0.0
        probs: list[float] = []
        for step_scores in scores:
            step_probs = torch.softmax(step_scores[0], dim=-1)
            probs.append(float(step_probs.max().item()))
        if not probs:
            return 0.0
        return float(np.mean(probs))

    def _candidate_rank(self, candidate: DigitResult) -> tuple[int, float, int]:
        valid = 1 if self._passes_rules(candidate.text) else 0
        length_bonus = 1 if 1 <= len(candidate.text) <= self.config.max_digits else 0
        shorter_is_better = -len(candidate.text) if candidate.text else -99
        return (valid, length_bonus, candidate.confidence, shorter_is_better)

    def _digit_token_ids(self) -> set[int]:
        vocab = self.processor.tokenizer.get_vocab()
        return {token_id for token, token_id in vocab.items() if token.isdigit()}

    def _disallowed_token_ids(self, allowed_token_ids: set[int]) -> list[int]:
        vocab = self.processor.tokenizer.get_vocab()
        special = {
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.bos_token_id,
        }
        return [
            token_id
            for token_id in vocab.values()
            if token_id not in allowed_token_ids and token_id not in special
        ]

    def _segment_digits(self, image_crop: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if image_crop.ndim == 3 else image_crop
        gray = self._extract_numeric_window(gray) if gray.ndim == 2 else gray
        if gray is None or gray.size == 0:
            return []
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments: list[tuple[int, np.ndarray]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 20:
                continue
            if h < max(8, gray.shape[0] * 0.25):
                continue
            if x == 0 and w < max(6, gray.shape[1] * 0.08):
                continue
            crop = binary[y : y + h, x : x + w]
            segments.append((x, crop))

        segments.sort(key=lambda item: item[0])
        return [crop for _, crop in segments[: self.config.max_digits]]

    def _prepare_cnn_input(self, digit_binary: np.ndarray) -> np.ndarray:
        h, w = digit_binary.shape[:2]
        side = max(h, w) + 8
        canvas = np.zeros((side, side), dtype=np.uint8)
        y_offset = (side - h) // 2
        x_offset = (side - w) // 2
        canvas[y_offset : y_offset + h, x_offset : x_offset + w] = digit_binary
        resized = cv2.resize(canvas, (28, 28))
        normalized = resized.astype(np.float32) / 255.0
        return normalized[None, None, :, :]

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        stable = logits - np.max(logits)
        exp = np.exp(stable)
        return exp / np.clip(np.sum(exp), a_min=1e-8, a_max=None)

    def _passes_rules(self, text: str) -> bool:
        if not text:
            return False

        if not re.fullmatch(r"\d{1,%d}" % self.config.max_digits, text):
            return False

        if len(text) > 1 and text.startswith("0"):
            return False

        return True


def resolve_cnn_path(models_dir: str | Path) -> str | None:
    path = Path(models_dir) / "svhn_digit_cnn.onnx"
    return str(path) if path.exists() else None
