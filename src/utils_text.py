# -*- coding: utf-8 -*-
import re, unicodedata, yaml
from typing import Dict, List

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("，", ",").replace("；", ";").replace("、", ",").replace("|", ",")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def load_synonyms(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            syn = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    mapping = {}
    for key, synonyms in syn.items():
        key = normalize_text(key)
        mapping[key] = key
        for s in (synonyms or []):
            mapping[normalize_text(s)] = key
    return mapping

def apply_synonyms(items: List[str], mapping: Dict[str,str]) -> List[str]:
    out = []
    seen = set()
    for it in items:
        it = normalize_text(it)
        canon = mapping.get(it, it)
        if canon and canon not in seen:
            out.append(canon); seen.add(canon)
    return out