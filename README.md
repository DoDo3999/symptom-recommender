# Agnos — Symptom Recommender (Baseline)
Install: `pip install -r requirements.txt`
Train: `python -m src.train_recommender --config config/config.yaml`
Eval: `python -m src.eval --config config/config.yaml --k 10`
Serve: `uvicorn src.serve_api:app --reload --host 0.0.0.0 --port 8000`