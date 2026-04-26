from __future__ import annotations

import argparse
import json
from pathlib import Path

from advanced_main_pipeline import AdvancedOCRPipeline, build_intermediate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch extract intermediate OCR records from many PDFs/images")
    parser.add_argument("--input-root", default=".", help="Root directory to search recursively")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern to match input files")
    parser.add_argument("--models-dir", default="models", help="Directory containing model weights")
    parser.add_argument("--output-dir", default="intermediate_batch_output", help="Directory to write outputs")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of files")
    parser.add_argument("--name-contains", default="", help="Optional substring filter on filename")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_inputs(root: Path, pattern: str, limit: int, name_contains: str) -> list[Path]:
    files = sorted(root.rglob(pattern))
    if name_contains:
        files = [path for path in files if name_contains in path.name]
    if limit > 0:
        files = files[:limit]
    return files


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    per_file_dir = output_dir / "per_file_json"
    ensure_dir(output_dir)
    ensure_dir(per_file_dir)

    pipeline = AdvancedOCRPipeline(models_dir=args.models_dir)
    input_files = collect_inputs(Path(args.input_root), args.glob, args.limit, args.name_contains)

    master_records: list[dict] = []
    failed_records: list[dict] = []

    for idx, input_path in enumerate(input_files):
        try:
            result = pipeline.run(input_path)
            payload = build_intermediate_payload(
                result,
                pipeline.candidate_master,
                pipeline.party_master,
            )
            payload["batch_index"] = idx
            payload["record_id"] = f"file_{idx:05d}"
            master_records.append(payload)

            file_name = f"{idx:05d}_{sanitize_filename(input_path.stem)}.json"
            (per_file_dir / file_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"index": idx, "status": "ok", "file": str(input_path)}, ensure_ascii=False))
        except Exception as exc:
            failed = {
                "batch_index": idx,
                "source_file": str(input_path),
                "error": str(exc),
            }
            failed_records.append(failed)
            print(json.dumps({"index": idx, "status": "failed", "file": str(input_path), "error": str(exc)}, ensure_ascii=False))

    (output_dir / "master_intermediate.json").write_text(
        json.dumps(master_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failed_files.json").write_text(
        json.dumps(failed_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "num_inputs": len(input_files),
                "num_success": len(master_records),
                "num_failed": len(failed_records),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_inputs": len(input_files),
                "num_success": len(master_records),
                "num_failed": len(failed_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def sanitize_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe[:80]


if __name__ == "__main__":
    main()
