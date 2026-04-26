from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    page_index: int = 1
    row_hint: int | None = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRTextResult:
    text: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DigitResult:
    text: str
    confidence: float
    source: str
    review_required: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RowExtraction:
    row_index: int
    name_box: BoundingBox | None
    score_box: BoundingBox | None
    name: str = ""
    name_confidence: float = 0.0
    score: str = ""
    score_confidence: float = 0.0
    score_source: str = "unresolved"
    review_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class PipelineResult:
    file: str
    page: int
    status: str
    rows: list[RowExtraction]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "page": self.page,
            "status": self.status,
            "rows": [row.to_dict() for row in self.rows],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
