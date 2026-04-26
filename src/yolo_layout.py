from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.pipeline_types import BoundingBox

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError("ultralytics is required for layout detection") from exc


@dataclass
class LayoutConfig:
    model_path: str
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.45
    row_merge_tolerance: float = 0.40
    min_name_confidence: float = 0.28
    min_score_confidence: float = 0.50
    dedupe_iou_threshold: float = 0.65
    min_name_width_ratio: float = 0.20
    min_score_width_ratio: float = 0.12


class YoloLayoutDetector:
    def __init__(self, config: LayoutConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        self.class_names = self._extract_names()

    def _extract_names(self) -> dict[int, str]:
        names = getattr(self.model.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}

    def detect(self, image: np.ndarray, page_index: int = 1) -> list[BoundingBox]:
        results = self.model.predict(
            source=image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        boxes: list[BoundingBox] = []
        result = results[0]
        if result.boxes is None:
            return boxes

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)
        for coords, conf, cls_idx in zip(xyxy, confs, clses):
            x1, y1, x2, y2 = [int(round(v)) for v in coords.tolist()]
            label = self.class_names.get(cls_idx, str(cls_idx))
            boxes.append(
                BoundingBox(
                    label=label,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(conf),
                    page_index=page_index,
                )
            )

        boxes = self._filter_boxes(boxes, image.shape[1])
        boxes.sort(key=lambda item: (item.y1, item.x1))
        return boxes

    def pair_rows(
        self,
        boxes: Iterable[BoundingBox],
        image_shape: tuple[int, ...] | None = None,
    ) -> list[tuple[BoundingBox | None, BoundingBox | None]]:
        name_boxes = sorted([box for box in boxes if box.label == "name"], key=lambda box: (box.y1, box.x1))
        score_boxes = sorted([box for box in boxes if box.label == "score"], key=lambda box: (box.y1, box.x1))
        if not name_boxes and not score_boxes:
            return []

        paired: list[tuple[BoundingBox | None, BoundingBox | None]] = []
        used_scores: set[int] = set()

        for name_box in name_boxes:
            best_idx = None
            best_delta = None
            for score_idx, score_box in enumerate(score_boxes):
                if score_idx in used_scores:
                    continue
                if score_box.x1 <= name_box.x2:
                    continue
                delta = abs(self._center_y(name_box) - self._center_y(score_box))
                tolerance = max(name_box.height, score_box.height) * self.config.row_merge_tolerance
                if delta <= tolerance and (best_delta is None or delta < best_delta):
                    best_idx = score_idx
                    best_delta = delta
            if best_idx is None:
                paired.append((name_box, None))
                continue
            score_box = score_boxes[best_idx]
            used_scores.add(best_idx)
            paired.append((name_box, score_box))

        if image_shape is not None:
            paired = self._fill_missing_score_boxes(
                paired=paired,
                score_boxes=score_boxes,
                image_shape=image_shape,
            )

        paired.sort(key=lambda row: self._row_sort_key(*row))
        for row_idx, row in enumerate(paired):
            for box in row:
                if box is not None:
                    box.row_hint = row_idx
        return paired

    def crop_box(self, image: np.ndarray, box: BoundingBox | None, padding: int = 2) -> np.ndarray | None:
        if box is None:
            return None
        h, w = image.shape[:2]
        x1 = max(0, box.x1 - padding)
        y1 = max(0, box.y1 - padding)
        x2 = min(w, box.x2 + padding)
        y2 = min(h, box.y2 + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    def _filter_boxes(self, boxes: list[BoundingBox], image_width: int) -> list[BoundingBox]:
        filtered: list[BoundingBox] = []
        for box in boxes:
            width_ratio = box.width / max(1, image_width)
            if box.label == "name":
                if box.confidence < self.config.min_name_confidence:
                    continue
                if width_ratio < self.config.min_name_width_ratio:
                    continue
                if box.x2 > int(image_width * 0.78):
                    continue
            elif box.label == "score":
                if box.confidence < self.config.min_score_confidence:
                    continue
                if width_ratio < self.config.min_score_width_ratio:
                    continue
                if box.x1 < int(image_width * 0.35):
                    continue
            filtered.append(box)

        deduped: list[BoundingBox] = []
        for label in ("name", "score"):
            kept: list[BoundingBox] = []
            label_boxes = sorted(
                [box for box in filtered if box.label == label],
                key=lambda item: item.confidence,
                reverse=True,
            )
            for candidate in label_boxes:
                if any(self._iou(candidate, existing) >= self.config.dedupe_iou_threshold for existing in kept):
                    continue
                kept.append(candidate)
            deduped.extend(kept)
        return deduped

    def _center_y(self, box: BoundingBox) -> float:
        return (box.y1 + box.y2) / 2.0

    def _fill_missing_score_boxes(
        self,
        paired: list[tuple[BoundingBox | None, BoundingBox | None]],
        score_boxes: list[BoundingBox],
        image_shape: tuple[int, ...],
    ) -> list[tuple[BoundingBox | None, BoundingBox | None]]:
        if not paired:
            return paired

        image_height, image_width = image_shape[:2]
        score_x1, score_x2, score_height = self._infer_score_column(paired, score_boxes, image_width)
        if score_x2 <= score_x1:
            return paired

        filled: list[tuple[BoundingBox | None, BoundingBox | None]] = []
        for name_box, score_box in paired:
            if score_box is not None or name_box is None:
                filled.append((name_box, score_box))
                continue

            target_height = max(int(round(score_height)), name_box.height)
            center_y = int(round(self._center_y(name_box)))
            y1 = max(0, center_y - target_height // 2)
            y2 = min(image_height, y1 + target_height)
            synthetic = BoundingBox(
                label="score",
                x1=max(name_box.x2 + 4, score_x1),
                y1=y1,
                x2=score_x2,
                y2=y2,
                confidence=0.01,
                page_index=name_box.page_index,
                row_hint=name_box.row_hint,
            )
            if synthetic.x2 - synthetic.x1 >= max(40, int(image_width * 0.08)):
                filled.append((name_box, synthetic))
            else:
                filled.append((name_box, score_box))
        return filled

    def _infer_score_column(
        self,
        paired: list[tuple[BoundingBox | None, BoundingBox | None]],
        score_boxes: list[BoundingBox],
        image_width: int,
    ) -> tuple[int, int, float]:
        valid_scores = [box for box in score_boxes if box.width > 0 and box.height > 0]
        if valid_scores:
            score_x1 = int(np.median([box.x1 for box in valid_scores]))
            score_x2 = int(np.median([box.x2 for box in valid_scores]))
            score_height = float(np.median([box.height for box in valid_scores]))
            return score_x1, score_x2, score_height

        name_boxes = [name_box for name_box, _ in paired if name_box is not None]
        if not name_boxes:
            return int(image_width * 0.60), int(image_width * 0.95), float(image_width * 0.04)

        median_name_x2 = int(np.median([box.x2 for box in name_boxes]))
        median_name_h = float(np.median([box.height for box in name_boxes]))
        score_x1 = max(int(image_width * 0.56), median_name_x2 + int(image_width * 0.015))
        score_x2 = int(image_width * 0.965)
        return score_x1, score_x2, median_name_h * 0.95

    def _row_sort_key(self, name_box: BoundingBox | None, score_box: BoundingBox | None) -> tuple[int, int]:
        candidate = name_box or score_box
        assert candidate is not None
        return (candidate.y1, candidate.x1)

    def _iou(self, a: BoundingBox, b: BoundingBox) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union_area = a.width * a.height + b.width * b.height - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


def find_model_path(models_dir: str | Path, filename: str) -> str:
    path = Path(models_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing layout model: {path}")
    return str(path)
