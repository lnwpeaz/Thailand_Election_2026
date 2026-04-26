from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from src.pipeline_types import OCRTextResult


@dataclass
class OnnxOCRConfig:
    det_model_path: str
    rec_model_path: str
    rec_config_path: str | None = None
    providers: list[str] | None = None
    rec_img_height: int = 48
    rec_img_width: int = 320


class OnnxThaiOCR:
    """
    Lightweight wrapper for PP-OCR style ONNX inference.

    This module intentionally keeps the detector integration simple:
    - detection session is loaded and validated
    - recognition session is used on the supplied crop directly
    - the pipeline is expected to provide cell-level crops from YOLO
    """

    def __init__(self, config: OnnxOCRConfig):
        self.config = config
        providers = config.providers or ort.get_available_providers()
        self.det_session = ort.InferenceSession(config.det_model_path, providers=providers)
        self.rec_session = ort.InferenceSession(config.rec_model_path, providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.vocab = self._load_vocab(config.rec_config_path)

    def recognize_crop(self, image_crop: np.ndarray | None) -> OCRTextResult:
        if image_crop is None or image_crop.size == 0:
            return OCRTextResult(text="", confidence=0.0, source="onnx_ocr")

        rec_tensor = self._prepare_rec_input(image_crop)
        outputs = self.rec_session.run(None, {self.rec_input_name: rec_tensor})
        text, confidence = self._decode_recognition(outputs)
        return OCRTextResult(text=text, confidence=confidence, source="onnx_ocr")

    def validate_detector(self, image: np.ndarray) -> dict[str, Any]:
        det_tensor = self._prepare_det_input(image)
        outputs = self.det_session.run(None, {self.det_input_name: det_tensor})
        return {
            "output_count": len(outputs),
            "shapes": [list(np.asarray(output).shape) for output in outputs],
        }

    def _prepare_det_input(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        img = cv2.resize(img, (960, 960))
        tensor = img.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        return tensor

    def _prepare_rec_input(self, image_crop: np.ndarray) -> np.ndarray:
        if image_crop.ndim == 2:
            color = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)
        else:
            color = image_crop
        target_h = self.config.rec_img_height
        scale = target_h / max(1, color.shape[0])
        target_w = min(self.config.rec_img_width, max(16, int(color.shape[1] * scale)))
        resized = cv2.resize(color, (target_w, target_h))
        canvas = np.full((target_h, self.config.rec_img_width, 3), 255, dtype=np.uint8)
        canvas[:, :target_w, :] = resized
        tensor = canvas.astype(np.float32) / 127.5 - 1.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        return tensor

    def _decode_recognition(self, outputs: list[np.ndarray]) -> tuple[str, float]:
        logits = np.asarray(outputs[0])
        if logits.ndim == 4:
            logits = logits.squeeze(0).squeeze(1)
        elif logits.ndim == 3:
            logits = logits.squeeze(0)
        if logits.ndim != 2:
            return "", 0.0

        probs = self._softmax(logits)
        indices = probs.argmax(axis=1).tolist()
        confidence = float(probs.max(axis=1).mean())

        text_chars: list[str] = []
        last_idx = None
        for idx in indices:
            if idx == 0 or idx == last_idx:
                last_idx = idx
                continue
            text_chars.append(self.vocab[idx] if idx < len(self.vocab) else "")
            last_idx = idx
        text = "".join(text_chars).strip()
        return text, confidence

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        stable = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(stable)
        return exp / np.clip(exp.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)

    def _build_default_thai_vocab(self) -> list[str]:
        digits = list("0123456789")
        latin = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        thai = [chr(code) for code in range(0x0E01, 0x0E5C)]
        punctuation = list(" -./(),:%")
        return [""] + digits + latin + thai + punctuation

    def _load_vocab(self, rec_config_path: str | None) -> list[str]:
        if rec_config_path:
            path = Path(rec_config_path)
            if path.exists():
                try:
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                    characters = payload.get("PostProcess", {}).get("character_dict")
                    if isinstance(characters, list) and characters:
                        return [""] + [str(item) for item in characters]
                except Exception:
                    pass
        return self._build_default_thai_vocab()


def resolve_onnx_paths(models_dir: str | Path) -> tuple[str, str, str | None]:
    models_path = Path(models_dir)
    det = models_path / "ppocr_v5_det.onnx"
    rec = models_path / "ppocr_v5_thai_rec.onnx"
    rec_yml = Path("paddle_models") / "th_PP-OCRv5_mobile_rec_infer" / "inference.yml"
    missing = [str(path) for path in (det, rec) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX OCR model(s): {', '.join(missing)}")
    return str(det), str(rec), str(rec_yml) if rec_yml.exists() else None
