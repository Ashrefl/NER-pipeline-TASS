"""
ÉTAPE 3 — Fine-tuning modèle NER spaCy
Entrée  : train.json + dev.json  (ou train_llm.json + dev_llm.json si tu as les annotations LLM)
Sortie  : model_ner/model-best/

Prérequis :
    pip install spacy
    # Si tu as les annotations LLM, renomme train_llm.json → train.json etc.

Exécuter :
    python 2_finetune_ner.py
"""

import json, random, os, spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import filter_spans

TRAIN_JSON = "train_llm.json" if os.path.exists("train_llm.json") else "train.json"
DEV_JSON   = "dev_llm.json"   if os.path.exists("dev_llm.json")   else "dev.json"
MODEL_OUT  = "model_ner"
EPOCHS     = 30
BATCH_SIZE = 16
DROPOUT    = 0.3

print(f"Données : {TRAIN_JSON} / {DEV_JSON}")

# ─── 1. Convertir JSON → DocBin ───────────────────────────────────────────────
def json_to_docs(json_path, nlp):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        doc = nlp.make_doc(item["text"])
        spans = []
        for start, end, label in item["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span:
                spans.append(span)
        doc.ents = filter_spans(spans)
        if doc.ents:
            docs.append(doc)
    return docs

# ─── 2. Modèle NER ────────────────────────────────────────────────────────────
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
for label in ("MIL_UNIT", "MIL_ORG", "MIL_WEAPON"):
    ner.add_label(label)

train_docs = json_to_docs(TRAIN_JSON, nlp)
dev_docs   = json_to_docs(DEV_JSON,   nlp)
print(f"Train : {len(train_docs)} docs | Dev : {len(dev_docs)} docs")

train_ex = [Example(nlp.make_doc(d.text), d) for d in train_docs]
dev_ex   = [Example(nlp.make_doc(d.text), d) for d in dev_docs]

# ─── 3. Entraînement ──────────────────────────────────────────────────────────
nlp.initialize(lambda: train_ex)
optimizer = nlp.resume_training()
os.makedirs(MODEL_OUT, exist_ok=True)

best_f = 0
patience = 0
MAX_PATIENCE = 8  # early stopping

print(f"\nEntraînement ({EPOCHS} epochs max)...")
print(f"{'Epoch':>6} {'Loss':>8} {'F':>7} {'P':>7} {'R':>7}")

for epoch in range(EPOCHS):
    random.shuffle(train_ex)
    losses = {}
    for batch in spacy.util.minibatch(train_ex, size=BATCH_SIZE):
        nlp.update(batch, sgd=optimizer, drop=DROPOUT, losses=losses)

    sc  = nlp.evaluate(dev_ex)
    f, p, r = sc["ents_f"], sc["ents_p"], sc["ents_r"]
    marker = ""

    if f > best_f:
        best_f = f
        patience = 0
        nlp.to_disk(f"{MODEL_OUT}/model-best")
        marker = " ← BEST"
    else:
        patience += 1

    print(f"{epoch+1:>6} {losses['ner']:>8.1f} {f:>7.3f} {p:>7.3f} {r:>7.3f}{marker}")

    if patience >= MAX_PATIENCE:
        print(f"\nEarly stopping (patience={MAX_PATIENCE})")
        break

# ─── 4. Résultats par label ───────────────────────────────────────────────────
nlp_best = spacy.load(f"{MODEL_OUT}/model-best")
best_ex  = [Example(nlp_best.make_doc(d.text), d) for d in dev_docs]
sc_final = nlp_best.evaluate(best_ex)

print(f"\n=== Résultats finaux (modèle best) ===")
print(f"F-score global : {sc_final['ents_f']:.3f}")
if "ents_per_type" in sc_final:
    for label, scores in sc_final["ents_per_type"].items():
        print(f"  {label:15} F={scores['f']:.3f}  P={scores['p']:.3f}  R={scores['r']:.3f}")

print(f"\n✓ Modèle sauvegardé dans {MODEL_OUT}/model-best/")
print("Prochaine étape : lance 3_inference.py")