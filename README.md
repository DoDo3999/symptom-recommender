# Agnos Symptom Recommender

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## Overview
ระบบแนะนำอาการ (Symptom Recommender) จากข้อมูลผู้ป่วย 1,000 ราย โดยเมื่อผู้ใช้เลือกอาการ ระบบจะแนะนำอาการถัดไปที่มักพบร่วมกัน  
โมเดลนี้ใช้ **co-occurrence + lift metric** พร้อมปรับตามเพศและอายุของผู้ใช้

---

## Quickstart
```bash
git clone https://github.com/DoDo3999/symptom-recommender.git
cd symptom-recommender
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Train model (ผู้ใช้ต้องใส่ไฟล์ data/raw/patients.csv เอง)
python -m src.train_recommender --config config/config.yaml

# Serve API
uvicorn src.serve_api:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI: http://127.0.0.1:8000/docs

---

## Project Structure
```
agnos_symptom_rec/
  config/
    config.yaml        # config parameters
    synonyms.yaml      # symptom synonyms
  data/
    raw/patients.csv   # <-- NOT INCLUDED (confidential)
  model/               # trained artifacts
    model.json
    vocab.json
  src/
    data_prep.py
    train_recommender.py
    eval.py
    serve_api.py
    utils_text.py
  requirements.txt
  README.md
```

---

## Training
```bash
python -m src.train_recommender --config config/config.yaml
```

## Evaluation
```bash
python -m src.eval --config config/config.yaml --k 10
```

### Evaluation Results
| Metric | Value |
|-------:|:-----|
| Hit@10 | 0.38 |
| MAP@10 | 0.20 |

---

## Serving API
```bash
uvicorn src.serve_api:app --reload --host 0.0.0.0 --port 8000
```
Swagger UI: http://localhost:8000/docs

---

## API Example

**Request**
```json
{
  "gender": "ชาย",
  "age": 26,
  "symptoms": ["ไอ"],
  "top_k": 10
}
```

**Response**
```json
{
  "bucket": "ชาย|||20-39",
  "items": [
    {"symptom": "น้ำมูกไหล", "score": 5.3},
    {"symptom": "เจ็บคอ", "score": 4.2}
  ]
}
```

---

## Notes
- ปรับ `symptom_text_cols` ใน `config.yaml` เพื่อเลือกว่าจะใช้ `search_term` หรือ `summary`
- ขยายคำพ้องได้ใน `synonyms.yaml`
- ⚠️ Repo นี้ไม่รวมไฟล์ข้อมูลจริง (confidential)  
  ผู้ใช้ต้องนำไฟล์ของตนเองมาใส่ที่: `data/raw/patients.csv`  
  Schema: `gender, age, search_term, summary`

---

## Future Work
- ใช้ embeddings (Sentence-BERT) แทน co-occurrence
- ขยาย `synonyms.yaml` ให้ครอบคลุมคำสะกดมากขึ้น
- Deploy API ด้วย Docker หรือบน Cloud (AWS/GCP/Azure)

---

## Contact
Author: DoDo3999  
For any questions, please open an issue or contact via GitHub.
