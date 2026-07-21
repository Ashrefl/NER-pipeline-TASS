"""
ÉTAPE 4 — Inférence NER sur tout le corpus (21 742 articles)
Entrée  : data_set.json  +  model_ner/model-best/
Sortie  : corpus_annotated.json

Exécuter :
    python 3_inference.py
"""

import json, spacy
from datetime import datetime

CORPUS_IN  = "data_set (1).json"   # fichier original complet
MODEL_PATH = "model_ner/model-best"
OUTPUT     = "corpus_annotated.json"
BATCH_SIZE = 64   # plus grand = plus rapide

print(f"Chargement du modèle : {MODEL_PATH}")
nlp = spacy.load(MODEL_PATH)

print(f"Chargement du corpus : {CORPUS_IN}")
with open(CORPUS_IN, encoding="utf-8") as f:
    articles = json.load(f)

print(f"{len(articles)} articles à traiter...")

# ─── Fonction de conversion date EPOCH → ISO ─────────────────────────────────
def epoch_to_iso(ts):
    if ts is None:
        return None
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        return None

# ─── Inférence par batch ──────────────────────────────────────────────────────
results = []
texts   = []
metas   = []

for art in articles:
    full = (art.get("title") or "") + " " + (art.get("text") or "")
    texts.append(full.strip())
    metas.append({
        "id":       art["id"],
        "title":    art.get("title", ""),
        "date_epoch": art.get("date"),
        "date_iso":   epoch_to_iso(art.get("date")),
        "tags":     art.get("tags", []),
        "mark":     art.get("mark"),
        "link":     art.get("link", ""),
        "isFlash":  art.get("isFlash", 0),
    })

# Inférence avec pipe (batched, plus rapide)
print(f"Inférence en cours (batch={BATCH_SIZE})...")
processed = 0
for doc, meta in zip(nlp.pipe(texts, batch_size=BATCH_SIZE), metas):
    entities = []
    for ent in doc.ents:
        entities.append({
            "text":  ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char
        })

    results.append({**meta, "entities": entities, "text": doc.text})
    processed += 1
    if processed % 2000 == 0:
        print(f"  {processed}/{len(articles)} traités...")

# ─── Sauvegarde ───────────────────────────────────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ─── Stats ────────────────────────────────────────────────────────────────────
total_ents = sum(len(r["entities"]) for r in results)
with_ents  = sum(1 for r in results if r["entities"])
by_label   = {"MIL_UNIT": 0, "MIL_ORG": 0, "MIL_WEAPON": 0}
for r in results:
    for e in r["entities"]:
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1

print(f"\n=== RÉSULTATS INFÉRENCE ===")
print(f"Articles traités      : {len(results)}")
print(f"Articles avec entités : {with_ents}")
print(f"Total entités         : {total_ents}")
print(f"  MIL_ORG    : {by_label['MIL_ORG']}")
print(f"  MIL_WEAPON : {by_label['MIL_WEAPON']}")
print(f"  MIL_UNIT   : {by_label['MIL_UNIT']}")
print(f"\n✓ Sauvegardé dans {OUTPUT}")
print("Prochaine étape : lance 4_ingest_elasticsearch.py")