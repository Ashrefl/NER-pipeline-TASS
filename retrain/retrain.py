"""
Réentraînement automatisé du modèle NER.

Principe « champion / challenger » :
  1. on entraîne un nouveau modèle (challenger) sur les données annotées à jour ;
  2. on mesure le modèle en production (champion) et le challenger sur le MÊME jeu de validation ;
  3. le challenger remplace le champion seulement s'il est meilleur ;
  4. chaque décision est tracée dans retrain/history.json.

Lancer depuis la racine du projet :
    python retrain/retrain.py
    python retrain/retrain.py --epochs 10      (essai plus rapide)

Lancé automatiquement par GitHub Actions : .github/workflows/retrain.yml
"""

import argparse
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import spacy

ROOT       = Path(__file__).resolve().parent.parent
PROD_MODEL = ROOT / "model_ner" / "model-best"
CANDIDATE  = ROOT / "model_ner_candidate"
ARCHIVE    = ROOT / "model_ner" / "archive"
HISTORY    = ROOT / "retrain" / "history.json"
MIN_GAIN   = 0.005   # gain minimal de F-score pour remplacer le modèle


def load_finetune():
    """Réutilise les fonctions de 2_finetune_ner.py, sans dupliquer le code."""
    path = ROOT / "2_finetune_ner.py"
    spec = importlib.util.spec_from_file_location("finetune", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(model_path, dev_json, ft):
    """F-score global d'un modèle sur le jeu de validation."""
    if not Path(model_path).exists():
        return 0.0
    nlp = spacy.load(model_path)
    docs = ft.json_to_docs(dev_json, nlp)
    return nlp.evaluate(ft.docs_to_examples(nlp, docs))["ents_f"]


def decide(f_prod, f_new, min_gain=MIN_GAIN):
    """Règle de décision : on déploie seulement si le gain est réel."""
    return f_new >= f_prod + min_gain


def save_history(entry):
    history = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else []
    history.append(entry)
    HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "train_llm.json"))
    parser.add_argument("--dev", default=str(ROOT / "dev_llm.json"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    ft = load_finetune()

    print("1/3 Entraînement du challenger...")
    nlp = ft.build_model()
    train_docs = ft.json_to_docs(args.train, nlp)
    dev_docs   = ft.json_to_docs(args.dev, nlp)
    print(f"    {len(train_docs)} exemples d'entraînement, {len(dev_docs)} de validation")
    ft.train(nlp, ft.docs_to_examples(nlp, train_docs), ft.docs_to_examples(nlp, dev_docs),
             model_out=str(CANDIDATE), epochs=args.epochs)

    print("2/3 Comparaison sur le même jeu de validation...")
    f_prod = evaluate(PROD_MODEL, args.dev, ft)
    f_new  = evaluate(CANDIDATE / "model-best", args.dev, ft)
    print(f"    Modèle en production : F = {f_prod:.3f}")
    print(f"    Nouveau modèle       : F = {f_new:.3f}")

    deploy = decide(f_prod, f_new)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    print("3/3 Décision :", "DÉPLOIEMENT du nouveau modèle" if deploy else "on GARDE le modèle actuel")

    if deploy:
        if PROD_MODEL.exists():
            ARCHIVE.mkdir(parents=True, exist_ok=True)
            shutil.copytree(PROD_MODEL, ARCHIVE / f"model-{stamp}")   # retour arrière possible
            shutil.rmtree(PROD_MODEL)
        shutil.copytree(CANDIDATE / "model-best", PROD_MODEL)

    save_history({
        "date": stamp,
        "train_examples": len(train_docs),
        "dev_examples": len(dev_docs),
        "f_production": round(f_prod, 4),
        "f_candidate": round(f_new, 4),
        "deployed": deploy,
    })
    shutil.rmtree(CANDIDATE, ignore_errors=True)
    print(f"✓ Décision enregistrée dans {HISTORY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
