"""
Ajout automatique de nouveaux exemples annotés, avant le réentraînement.

1. Lit les nouveaux articles collectés par 0_scrape_tass.py.
2. Écarte les articles trop courts et ceux déjà annotés.
3. Les fait annoter par Mistral, avec les mêmes fonctions que 1_annotate_llm.py.
4. Ajoute ceux qui contiennent au moins une entité à train_llm.json.
   dev_llm.json ne change pas : l'examen reste le même pour comparer
   honnêtement l'ancien et le nouveau modèle.

Lancer depuis la racine du projet :
    python retrain/add_new_examples.py run/data_set.json
    python retrain/add_new_examples.py run/data_set.json --max 30

La clé Mistral est lue dans la variable MISTRAL_API_KEY
(fichier .env en local, secret GitHub dans le workflow retrain.yml).
"""

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
TRAIN       = ROOT / "train_llm.json"
DEV         = ROOT / "dev_llm.json"
IDS_FILE    = ROOT / "retrain" / "annotated_ids.json"
NEW_RECORDS = ROOT / "retrain" / "new_annotations.json"
MIN_CHARS   = 100   # même filtre que pour les 500 premiers articles
BATCH       = 3     # même taille de lot que 1_annotate_llm.py
DELAY       = 2.0   # pause entre deux appels à Mistral
RETRY_WAITS = [10, 30, 0]   # 3 essais : attente de 10 s, puis 30 s, en cas de refus de Mistral


def load_annotator():
    """Réutilise les fonctions de 1_annotate_llm.py, sans dupliquer le code."""
    spec = importlib.util.spec_from_file_location("annotate", ROOT / "1_annotate_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path, default):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def full_text(article):
    return ((article.get("title") or "") + " " + (article.get("text") or "")).strip()


def select_new(articles, known_ids, known_texts, max_articles):
    """Garde les articles assez longs, jamais annotés, sans doublon."""
    selected, seen = [], set()
    for a in articles:
        text = full_text(a)
        if len(a.get("text") or "") <= MIN_CHARS:
            continue
        if a.get("id") in known_ids or a.get("id") in seen or text in known_texts:
            continue
        seen.add(a.get("id"))
        selected.append(a)
    return selected[:max_articles]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("new_articles", help="fichier des articles collectés, ex. run/data_set.json")
    p.add_argument("--max", type=int, default=30, help="nombre maximum d'articles à annoter")
    args = p.parse_args()

    if not os.getenv("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY absente : aucun nouvel exemple ajouté.")
        return

    articles = read_json(args.new_articles, [])
    train = read_json(TRAIN, [])
    dev = read_json(DEV, [])
    known_ids = set(read_json(IDS_FILE, []))
    known_texts = {ex["text"] for ex in train + dev}

    batch_articles = select_new(articles, known_ids, known_texts, args.max)
    print(f"{len(articles)} articles collectés, {len(batch_articles)} nouveaux à annoter")
    if not batch_articles:
        return

    ann = load_annotator()
    added, records = 0, read_json(NEW_RECORDS, [])
    for i in range(0, len(batch_articles), BATCH):
        group = batch_articles[i:i + BATCH]
        results = None
        for attempt, wait in enumerate(RETRY_WAITS, start=1):
            try:
                results = ann.annotate_batch(group)
                break
            except Exception as e:
                if attempt == len(RETRY_WAITS):
                    print(f"  Lot ignoré après {attempt} essais, erreur Mistral : {e}")
                else:
                    print(f"  Erreur Mistral ({e}), nouvel essai dans {wait} s...")
                    time.sleep(wait)
        if results is None:
            continue   # ces articles seront retentés à la prochaine exécution
        for art, raw_entities in zip(group, results):
            text = full_text(art)
            spans = ann.find_spans(text, raw_entities)
            known_ids.add(art.get("id"))
            if spans:
                train.append({"text": text, "entities": [[s["start"], s["end"], s["label"]] for s in spans]})
                records.append({"id": art.get("id"), "date": art.get("date"),
                                "annotated_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                "entities": spans})
                added += 1
        time.sleep(DELAY)

    write_json(TRAIN, train)
    write_json(IDS_FILE, sorted(i for i in known_ids if i is not None))
    write_json(NEW_RECORDS, records)
    print(f"✓ {added} nouveaux exemples ajoutés à train_llm.json ({len(train)} au total)")


if __name__ == "__main__":
    main()
