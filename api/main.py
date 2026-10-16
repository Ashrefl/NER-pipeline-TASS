"""
API de prédiction du modèle NER militaire (serving).

Lancer en local, depuis la racine du projet :
    uvicorn api.main:app --reload
Puis ouvrir la documentation interactive : http://localhost:8000/docs

Routes :
    GET  /health         état de l'API et du modèle chargé
    POST /predict        entités trouvées dans un texte
    POST /predict/batch  entités trouvées dans plusieurs textes
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model_ner/model-best")
LOG_PATH   = Path(os.getenv("PREDICTION_LOG", "monitoring/predictions_log.jsonl"))
LABELS     = ("MIL_UNIT", "MIL_ORG", "MIL_WEAPON")
MAX_CHARS  = 20_000   # limite la taille d'un texte pour protéger l'API

app = FastAPI(
    title="API NER militaire — dépêches TASS",
    description="Repère les unités (MIL_UNIT), organisations (MIL_ORG) et armes (MIL_WEAPON).",
    version="1.0.0",
)

_nlp = None


def get_model():
    """Charge le modèle une seule fois, au premier appel."""
    global _nlp
    if _nlp is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(status_code=503, detail=f"Modèle introuvable : {MODEL_PATH}")
        _nlp = spacy.load(MODEL_PATH)
    return _nlp


# ─── Formats d'entrée et de sortie ────────────────────────────────────────────
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS,
                      examples=["Russia deployed S-400 systems near the border, the Defense Ministry said."])


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


def extract_entities(doc):
    """Transforme un document spaCy en liste d'entités lisibles."""
    return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
            for e in doc.ents]


def count_labels(entities):
    counts = {label: 0 for label in LABELS}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return counts


def log_prediction(text, entities, duration_ms):
    """
    Journal de production : une ligne par prédiction, sans stocker le texte.
    Ce journal alimente le rapport de dérive (monitoring/drift_report.py).
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "n_chars":     len(text),
            "n_entities":  len(entities),
            **{f"n_{k.lower()}": v for k, v in count_labels(entities).items()},
            "duration_ms": round(duration_ms, 1),
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass   # le journal ne doit jamais bloquer une prédiction


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Vérifie que l'API répond et que le modèle se charge."""
    nlp = get_model()
    labels = list(nlp.get_pipe("ner").labels) if "ner" in nlp.pipe_names else []
    return {"status": "ok", "model": MODEL_PATH, "pipeline": nlp.pipe_names, "labels": labels}


@app.post("/predict")
def predict(item: TextIn):
    nlp = get_model()
    start = time.perf_counter()
    entities = extract_entities(nlp(item.text))
    duration = (time.perf_counter() - start) * 1000
    log_prediction(item.text, entities, duration)
    return {"entities": entities, "counts": count_labels(entities), "duration_ms": round(duration, 1)}


@app.post("/predict/batch")
def predict_batch(items: BatchIn):
    nlp = get_model()
    results = []
    for text, doc in zip(items.texts, nlp.pipe(items.texts, batch_size=32)):
        entities = extract_entities(doc)
        log_prediction(text, entities, 0)
        results.append({"entities": entities, "counts": count_labels(entities)})
    return {"results": results}
