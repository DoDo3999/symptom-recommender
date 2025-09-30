# -*- coding: utf-8 -*-
import re, yaml, pandas as pd
from typing import List, Dict, Any
from .utils_text import normalize_text, load_synonyms, apply_synonyms

def _choose_first_existing(df, cands: List[str]):
    out = []
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            out.append(low[c.lower()])
    return out

def detect_columns(df: pd.DataFrame, cfg: Dict[str,Any]):
    hints = cfg.get("hints", {})
    gender_cols = _choose_first_existing(df, hints.get("gender_cols", []))
    age_cols = _choose_first_existing(df, hints.get("age_cols", []))
    sym_txt_cols = _choose_first_existing(df, hints.get("symptom_text_cols", []))
    one_hot_prefixes = hints.get("one_hot_prefixes", ["symptom_", "sx_"])
    delims = hints.get("delimiter_candidates", [",",";"])
    if not sym_txt_cols:
        sym_txt_cols = _choose_first_existing(df, ["symptoms","symptom","อาการ","summary"])
    one_hot_cols = []
    if not sym_txt_cols:
        for c in df.columns:
            lc = c.lower()
            if any(lc.startswith(p) for p in one_hot_prefixes):
                vs = pd.to_numeric(df[c], errors="coerce").dropna().unique().tolist()
                if set(vs).issubset({0,1,0.0,1.0}):
                    one_hot_cols.append(c)
    return {"gender_cols": gender_cols, "age_cols": age_cols, "symptom_text_cols": sym_txt_cols, "one_hot_cols": one_hot_cols, "delims": delims}

def parse_symptoms_row(row, det: Dict[str,Any], synmap: Dict[str,str]):
    sx = []
    for col in det["symptom_text_cols"]:
        text = normalize_text(row.get(col, ""))
        if not text: continue
        parts = re.split(r"[,\;\|/]+", text)
        sx.extend([p.strip() for p in parts if p.strip()])
    for col in det["one_hot_cols"]:
        v = row.get(col, 0)
        try: v = float(v)
        except: v = 0
        if v == 1:
            label = re.sub(r"^(symptom_|sx_)", "", col, flags=re.IGNORECASE).replace("_", " ").strip()
            sx.append(label)
    return apply_synonyms(sx, synmap)

def normalize_gender(x) -> str:
    t = normalize_text(x).lower()
    if t in ["ชาย","m","male","man"]: return "ชาย"
    if t in ["หญิง","f","female","woman"]: return "หญิง"
    return "ไม่ระบุ"

def age_to_group(age, bins: List[List[int]]):
    try: a = int(float(age))
    except: return "NA"
    for lo,hi in bins:
        if lo <= a <= hi: return f"{lo}-{hi}"
    return "NA"

def build_transactions(csv_path: str, cfg: Dict[str,Any]):
    df = pd.read_csv(csv_path)
    det = detect_columns(df, cfg)
    synmap = load_synonyms(cfg.get("synonyms_path",""))
    bins = cfg.get("age_bins", [[0,12],[13,19],[20,39],[40,59],[60,200]])
    rows = []
    for _, r in df.iterrows():
        sx = parse_symptoms_row(r, det, synmap)
        if not sx: continue
        g = "ไม่ระบุ"
        for c in det["gender_cols"]:
            if pd.notna(r.get(c)): g = r.get(c); break
        a = "NA"
        for c in det["age_cols"]:
            if pd.notna(r.get(c)): a = r.get(c); break
        rows.append({"gender": normalize_gender(g), "age_group": age_to_group(a, bins), "symptoms": sx})
    return pd.DataFrame(rows), det