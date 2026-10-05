"""
ÉTAPE 3 — Fine-tuning du modèle NER spaCy (code de développement)
Entrée  : train_llm.json + dev_llm.json  (produits par 1_annotate_llm.py)
Sortie  : model_ner/model-best/          (meilleur modèle selon le F-score)

Prérequis :
    pip install spacy
Exécuter :
    python 2_finetune_ner.py
"""

import json
import os
import random

import spacy
from spacy.training import Example
from spacy.util import filter_spans, minibatch

TRAIN_JSON   = "train_llm.json"
DEV_JSON     = "dev_llm.json"
MODEL_OUT    = "model_ner"
LABELS       = ("MIL_UNIT", "MIL_ORG", "MIL_WEAPON")
EPOCHS       = 30     # nombre maximal de passes sur les données
BATCH_SIZE   = 16     # exemples par mise à jour
DROPOUT      = 0.3    # régularisation contre le sur-apprentissage
MAX_PATIENCE = 8      # early stopping : epochs sans amélioration avant arrêt
SEED         = 42     # graine fixe : entraînement reproductible


# ─── Préparation des données ──────────────────────────────────────────────────
def json_to_docs(json_path, nlp):
    """
    Convertit un fichier annoté [{text, entities: [[start, end, label]]}]
    en documents spaCy. Les entités mal alignées sur les mots sont ignorées
    et les chevauchements résolus par filter_spans.
    """
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


def docs_to_examples(nlp, docs):
    """Associe chaque texte brut à sa version annotée (format d'entraînement spaCy)."""
    return [Example(nlp.make_doc(d.text), d) for d in docs]


# ─── Modèle ───────────────────────────────────────────────────────────────────
def build_model():
    """Modèle vide en anglais avec un composant NER et les 3 labels militaires."""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in LABELS:
        ner.add_label(label)
    return nlp


# ─── Entraînement ─────────────────────────────────────────────────────────────
def train(nlp, train_ex, dev_ex, model_out=MODEL_OUT, epochs=EPOCHS):
    """
    Entraîne le modèle et sauvegarde le meilleur selon le F-score de validation.
    S'arrête si aucune amélioration pendant MAX_PATIENCE epochs (early stopping).
    Retourne le meilleur F-score obtenu.
    """
    random.seed(SEED)
    nlp.initialize(lambda: train_ex)
    optimizer = nlp.resume_training()
    os.makedirs(model_out, exist_ok=True)

    best_f, patience = 0.0, 0
    print(f"{'Epoch':>6} {'Loss':>8} {'F':>7} {'P':>7} {'R':>7}")

    for epoch in range(epochs):
        random.shuffle(train_ex)
        losses = {}
        for batch in minibatch(train_ex, size=BATCH_SIZE):
            nlp.update(batch, sgd=optimizer, drop=DROPOUT, losses=losses)

        scores = nlp.evaluate(dev_ex)
        f, p, r = scores["ents_f"], scores["ents_p"], scores["ents_r"]
        marker = ""
        if f > best_f:
            best_f, patience = f, 0
            nlp.to_disk(f"{model_out}/model-best")
            marker = " ← BEST"
        else:
            patience += 1

        print(f"{epoch + 1:>6} {losses.get('ner', 0):>8.1f} {f:>7.3f} {p:>7.3f} {r:>7.3f}{marker}")

        if patience >= MAX_PATIENCE:
            print(f"\nEarly stopping (patience={MAX_PATIENCE})")
            break
    return best_f


# ─── Évaluation finale ────────────────────────────────────────────────────────
def evaluate_best(model_path, dev_docs):
    """Recharge le meilleur modèle et affiche le F-score global et par label."""
    nlp_best = spacy.load(model_path)
    scores = nlp_best.evaluate(docs_to_examples(nlp_best, dev_docs))
    print("\n=== Résultats finaux (modèle best) ===")
    print(f"F-score global : {scores['ents_f']:.3f}")
    for label, s in (scores.get("ents_per_type") or {}).items():
        print(f"  {label:15} F={s['f']:.3f}  P={s['p']:.3f}  R={s['r']:.3f}")
    return scores


def main():
    print(f"Données : {TRAIN_JSON} / {DEV_JSON}")
    nlp = build_model()
    train_docs = json_to_docs(TRAIN_JSON, nlp)
    dev_docs   = json_to_docs(DEV_JSON, nlp)
    print(f"Train : {len(train_docs)} docs | Dev : {len(dev_docs)} docs")

    print(f"\nEntraînement ({EPOCHS} epochs max)...")
    train(nlp, docs_to_examples(nlp, train_docs), docs_to_examples(nlp, dev_docs))

    evaluate_best(f"{MODEL_OUT}/model-best", dev_docs)
    print(f"\n✓ Modèle sauvegardé dans {MODEL_OUT}/model-best/")
    print("Prochaine étape : lance 3_inference.py")


if __name__ == "__main__":
    main()
