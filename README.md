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
ง่ายมาก เดี๋ยวไล่ให้แบบ step-by-step เลย 👍

✅ วิธีเอา README ไปใส่ในโปรเจค
🔹 วิธีที่ 1 (ง่ายสุดใน VS Code)
1. สร้างไฟล์

ไปที่ folder project → คลิกขวา → New File
ตั้งชื่อ:

README.md
2. เปิดไฟล์ แล้ว paste
เอา README ที่ผมเขียนให้
Ctrl + V ใส่ลงไปเลย
3. Save
Ctrl + S
4. commit + push
git add README.md
git commit -m "add README"
git push
🔹 วิธีที่ 2 (ผ่าน Terminal)
notepad README.md

หรือ

code README.md

แล้ว paste เหมือนกัน

🔍 เช็คว่าโอเคมั้ย

ไปที่ GitHub repo จะเห็น:

👉 README โผล่หน้าแรกเลย
👉 มี format สวย ๆ (markdown)

💡 ทริคเล็ก ๆ (ทำให้ดูโปรขึ้น)
1. ใส่หัวข้อใหญ่ให้เด่น
# Thailand Election OCR Pipeline
2. ใส่ emoji (คุณมีแล้ว 👍)

GitHub render สวยขึ้น

3. Preview ใน VS Code

กด:

Ctrl + Shift + V
🚀 สรุป

ทำแค่:

สร้าง README.md
paste
save
git push

จบเลย 🔥

ถ้าคุณอยากให้ README:

ดู “เทพขึ้นอีกระดับ”
ใส่ diagram pipeline
เขียน wording สำหรับ HR

บอกได้ เดี๋ยวผมจัดเวอร์ชัน สมัคร intern ได้เลย ให้ 💯

ขอแบบก้อปวางใส่ readme

จัดให้แบบ copy → paste ลง README.md ได้เลย 👇

# 🇹🇭 Thailand Election OCR Pipeline

This project focuses on extracting vote scores from Thai election forms using OCR and structured data mapping.

The pipeline converts unstructured election documents into structured data that can be used for downstream machine learning or analytics tasks.

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
- Reads handwritten vote scores from images
- Supports:
  - Arabic digits (0–9)
  - Thai digits (๐–๙)
  - Thai number words (e.g., "หนึ่ง", "สิบเจ็ด")

---

### 2. Image ↔ Manifest Mapping
Each image is linked using:
- `record_id`
- `row_index`

Mapping is defined in:


score_annotation_export/score_annotation_manifest.json


---

### 3. Smart OCR Strategy
- Focuses on **left-most region** (actual handwritten score)
- Uses Thai word region (in parentheses) as fallback
- Handles edge cases:
  - Ambiguous digits: `0, 1, 4, 7`
  - Blank cells
  - Noisy handwriting

---

## 📂 Project Structure


project/
├── score_ocr/
│ ├── client.py
│ ├── utils.py
│ └── config.py
│
├── run_score_ocr.py
│
├── score_annotation_export/
│ ├── score_annotation_manifest.json
│ └── sample_score_crops/
│
├── score_ocr_results.csv
├── requirements.txt
└── README.md


---

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt


---

### 2. Start Ollama

ollama run qwen2.5vl:7b


---

### 3. Run OCR

python run_score_ocr.py --limit 100


---

## 📊 Output

The script generates:


score_ocr_results.csv


Key fields:
- `ocr_text` → predicted digits
- `final_score` → selected score
- `need_review` → whether manual review is required
- `review_reason` → reason for review

---

## ⚠️ Notes

- This repo includes only a **small sample dataset (~100 images)**
- Full dataset is not included due to size
- To run full pipeline, replace:


score_annotation_export/sample_score_crops/


with your dataset

---

## 🧩 Pipeline Context


PDF → Layout Detection → Row Extraction → Score Crop → OCR (this repo) → Model


---

## 🎯 Key Challenges

- OCR on low-quality handwritten Thai data
- Distinguishing similar digits (0, 1, 4, 7)
- Combining vision + language cues
- Reducing unnecessary manual review

---

## 👤 Author

Built as part of a data extraction and OCR pipeline for Thai election documents.
📦 requirements.txt (copy อันนี้ไปใส่อีกไฟล์)
opencv-python
numpy
pandas
requests
tqdm
✅ เสร็จแล้วทำต่อ
git add README.md requirements.txt
git commit -m "add README and requirements"
git push

ถ้าจะเอาแบบ โปรขึ้นอีก (ใส่ diagram / portfolio / resume bullet)
บอกได้ เดี๋ยวผมอัปเกรดให้ 🔥

เขียนสั้นๆพอให้เข้าใจ อันนี้คือทำไร มีโครงสร้างยังไง แล้วทำจะเอาไปทำ หาinsight ต่อ ต้องรันยังไง

ได้เลย อันนี้แบบ สั้น กระชับ เข้าใจง่าย เอาไปใส่ README ได้เลย 👇

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

pip install -r requirements.txt


2. Run model (Ollama)

ollama run qwen2.5vl:7b


3. Run OCR

python run_score_ocr.py


---

## 📊 Output
- `score_ocr_results.csv` → structured vote scores per row
