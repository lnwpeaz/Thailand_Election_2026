# Thailand Election OCR (Score Extraction)

This project extracts vote scores from Thai election forms using OCR.

## 📌 What this does
- Takes cropped score images from election documents
- Uses a vision-language model (Qwen2.5-VL) to read handwritten scores
- Maps each image to structured data using a manifest file
- Outputs results in CSV format for further analysis

---

## 📂 Structure


score_ocr/ # OCR logic
run_score_ocr.py # main script

score_annotation_export/
├── score_annotation_manifest.json # mapping image ↔ record
└── sample_score_crops/ # sample images

score_ocr_results.csv # output

---

## 🎯 Use case
The output can be used to:
- Analyze vote distribution
- Detect anomalies in election data
- Build ML models for prediction or validation

---

## 🚀 How to run
1. Install dependencies
```
pip install -r requirements.txt
```

2. Run model (Ollama)
```
ollama run qwen2.5vl:7b
```

3. Run OCR
```
python run_score_ocr.py
```

---

## 📊 Output
- `score_ocr_results.csv` → structured vote scores per row
