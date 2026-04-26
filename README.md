# 🇹🇭 Thailand Election OCR Pipeline

This project focuses on extracting vote scores from Thai election forms using OCR and structured data mapping.

The pipeline is designed to convert unstructured PDF election documents into structured data that can be used for downstream machine learning tasks.

---

## 📌 Project Scope

This repository covers the **score OCR stage** of the pipeline, including:

- Mapping cropped score images to structured metadata
- Running OCR on handwritten vote scores
- Combining OCR outputs with manifest data

---

## 🧠 What We Built

### 1. OCR for Vote Scores
- Uses **Qwen2.5-VL (Vision-Language Model)** via Ollama
- Reads handwritten digits from score cells
- Handles:
  - Arabic digits (0-9)
  - Thai digits (๐-๙)
  - Thai number words (e.g., "หนึ่ง", "สิบเจ็ด")

---

### 2. Image ↔ Manifest Mapping
- Each score image is linked via:
  - `record_id`
  - `row_index`
- Mapping is handled through:

```text
score_annotation_export/score_annotation_manifest.json

📂Project Structure
project/
 ├── score_ocr/                      # OCR logic
 │   ├── client.py                  # Qwen OCR calls
 │   ├── utils.py                   # image preprocessing + cropping
 │   └── config.py                  # config + thresholds
 │
 ├── run_score_ocr.py               # main OCR pipeline
 │
 ├── score_annotation_export/
 │   ├── score_annotation_manifest.json
 │   └── sample_score_crops/        # sample images (100 items)
 │
 ├── score_ocr_results.csv          # OCR output
 ├── requirements.txt
 └── README.md
🚀How to Run
1. Install dependencies
pip install -r requirements.txt
2. Start Ollama + Model
ollama run qwen2.5vl:7b
