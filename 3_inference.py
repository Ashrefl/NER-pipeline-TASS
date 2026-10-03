"""
ÉTAPE 4 — Inférence NER sur tout le corpus (21 742 articles)
Entrée  : data_set.json  +  model_ner/model-best/
Sortie  : corpus_annotated.json

Exécuter :
    python 3_inference.py
"""

import json
from datetime import datetime, timezone

import spacy

CORPUS_IN  = "data_set.json"       # corpus complet (sortie du scraper)
MODEL_PATH = "model_ner/model-best"
OUTPUT     = "corpus_annotated.json"
BATCH_SIZE = 64   # plus grand = plus rapide
LABELS     = ("MIL_UNIT", "MIL_ORG", "MIL_WEAPON")


# ─── Conversion date EPOCH → ISO 8601 ─────────────────────────────────────────
def epoch_to_iso(ts):
    """Convertit un timestamp EPOCH en date ISO ; None si absent ou invalide."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def build_text(art):
    """Texte analysé par le modèle : titre + corps (comme lors de l'annotation)."""
    return ((art.get("title") or "") + " " + (art.get("text") or "")).strip()


def build_meta(art):
    """Métadonnées conservées pour chaque article enrichi."""
    return {
        "id":         art["id"],
        "title":      art.get("title", ""),
        "date_epoch": art.get("date"),
        "date_iso":   epoch_to_iso(art.get("date")),
        "tags":       art.get("tags", []),
        "mark":       art.get("mark"),
        "link":       art.get("link", ""),
        "isFlash":    art.get("isFlash", 0),
    }


def doc_to_entities(doc):
    """Extrait les entités détectées par spaCy avec leur position dans le texte."""
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]


def count_by_label(results):
    """Compte les entités par label sur l'ensemble des articles enrichis."""
    counts = {label: 0 for label in LABELS}
    for r in results:
        for e in r["entities"]:
            counts[e["label"]] = counts.get(e["label"], 0) + 1
    return counts


# ─── Inférence par batch ──────────────────────────────────────────────────────
def annotate_corpus(nlp, articles, batch_size=BATCH_SIZE):
    """Applique le modèle à tous les articles avec nlp.pipe (traitement par lots)."""
    texts = [build_text(a) for a in articles]
    metas = [build_meta(a) for a in articles]

    results = []
    for doc, meta in zip(nlp.pipe(texts, batch_size=batch_size), metas):
        results.append({**meta, "entities": doc_to_entities(doc), "text": doc.text})
        if len(results) % 2000 == 0:
            print(f"  {len(results)}/{len(articles)} traités...")
    return results


def main():
    print(f"Chargement du modèle : {MODEL_PATH}")
    nlp = spacy.load(MODEL_PATH)

    print(f"Chargement du corpus : {CORPUS_IN}")
    with open(CORPUS_IN, encoding="utf-8") as f:
        articles = json.load(f)
    print(f"{len(articles)} articles à traiter...")

    print(f"Inférence en cours (batch={BATCH_SIZE})...")
    results = annotate_corpus(nlp, articles)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_ents = sum(len(r["entities"]) for r in results)
    with_ents  = sum(1 for r in results if r["entities"])
    by_label   = count_by_label(results)

    print(f"\n=== RÉSULTATS INFÉRENCE ===")
    print(f"Articles traités      : {len(results)}")
    print(f"Articles avec entités : {with_ents}")
    print(f"Total entités         : {total_ents}")
    print(f"  MIL_ORG    : {by_label['MIL_ORG']}")
    print(f"  MIL_WEAPON : {by_label['MIL_WEAPON']}")
    print(f"  MIL_UNIT   : {by_label['MIL_UNIT']}")
    print(f"\n✓ Sauvegardé dans {OUTPUT}")
    print("Prochaine étape : lance 4_ingest_elasticsearch.py")


if __name__ == "__main__":
    main()
