# Agnos Symptom Recommender

## Overview
ระบบแนะนำอาการ (Symptom Recommender) จากข้อมูลผู้ป่วย 1,000 ราย โดยเมื่อผู้ใช้เลือกอาการ ระบบจะแนะนำอาการถัดไปที่มักพบร่วมกัน  
โมเดลนี้ใช้ **co-occurrence + lift metric** พร้อมปรับตามเพศและอายุของผู้ใช้

## Project Structure
```
agnos_symptom_rec/
  config/
    config.yaml        # config parameters
    synonyms.yaml      # symptom synonyms
  data/
    raw/patients.csv   # input dataset
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

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
python -m src.train_recommender --config config/config.yaml
```

## Evaluation
```bash
python -m src.eval --config config/config.yaml --k 10
```

## Serving API
```bash
uvicorn src.serve_api:app --reload --host 0.0.0.0 --port 8000
```
Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

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

## Notes
- ปรับ `symptom_text_cols` ใน `config.yaml` เพื่อเลือกว่าจะใช้ `search_term` หรือ `summary`
- ขยายคำพ้องได้ใน `synonyms.yaml`
- Metrics baseline: Hit@10 ≈ 0.38, MAP@10 ≈ 0.20

** ข้อมูลผู้ป่วยเป็นความลับ — repo นี้ไม่รวมไฟล์จริง
โปรดวางไฟล์ของคุณเองที่: data/raw/patients.csv (ตาม schema: gender, age, search_term, summary)
