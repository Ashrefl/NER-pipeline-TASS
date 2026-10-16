"""
Monitoring du modèle : détection de dérive des données avec Evidently.

Idée : on compare une période de référence à une période récente.
Si les articles changent (longueur, nombre d'entités, sujets), le modèle
travaille sur des données différentes de celles qu'il connaît : c'est une dérive,
et un signal pour relancer le réentraînement (retrain/retrain.py).

Deux modes :
    python monitoring/drift_report.py
        compare 2016-2020 (référence) à 2021-février 2022 (période récente)
        à partir de corpus_annotated.json, la sortie de 3_inference.py
    python monitoring/drift_report.py --logs
        compare la première et la seconde moitié du journal de l'API
        (monitoring/predictions_log.jsonl)

Sortie : monitoring/drift_report.html, à ouvrir dans un navigateur.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT     = Path(__file__).resolve().parent.parent
CORPUS   = ROOT / "corpus_annotated.json"
LOGS     = ROOT / "monitoring" / "predictions_log.jsonl"
OUT_HTML = ROOT / "monitoring" / "drift_report.html"
LABELS   = ("MIL_UNIT", "MIL_ORG", "MIL_WEAPON")


def article_features(art):
    """Résume un article en quelques indicateurs numériques."""
    text = art.get("text") or ""
    ents = art.get("entities") or []
    feats = {
        "n_words": len(text.split()),
        "n_entities": len(ents),
        "mentions_ukraine": int("ukrain" in text.lower()),
    }
    for label in LABELS:
        feats[f"n_{label.lower()}"] = sum(1 for e in ents if e["label"] == label)
    return feats


def corpus_frames(path, ref_end, cur_start, cur_end):
    """Découpe le corpus annoté en période de référence et période récente."""
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)
    rows = []
    for a in articles:
        if not a.get("date_iso"):
            continue
        rows.append({"date": a["date_iso"][:10], **article_features(a)})
    df = pd.DataFrame(rows)
    ref = df[df["date"] <= ref_end].drop(columns="date")
    cur = df[(df["date"] >= cur_start) & (df["date"] <= cur_end)].drop(columns="date")
    return ref, cur


def log_frames(path):
    """Découpe le journal de l'API en deux moitiés : avant et maintenant."""
    df = pd.read_json(path, lines=True).drop(columns=["timestamp"], errors="ignore")
    half = len(df) // 2
    return df.iloc[:half], df.iloc[half:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", action="store_true", help="utiliser le journal de l'API")
    p.add_argument("--ref-end", default="2020-12-31")
    p.add_argument("--cur-start", default="2021-01-01")
    p.add_argument("--cur-end", default="2022-02-28")
    args = p.parse_args()

    if args.logs:
        ref, cur = log_frames(LOGS)
        label = "journal de l'API"
    else:
        ref, cur = corpus_frames(CORPUS, args.ref_end, args.cur_start, args.cur_end)
        label = f"référence jusqu'au {args.ref_end}, période récente {args.cur_start} → {args.cur_end}"

    if len(ref) < 10 or len(cur) < 10:
        sys.exit("Pas assez de données pour comparer les deux périodes.")

    print(f"Comparaison ({label}) : {len(ref)} lignes de référence, {len(cur)} récentes")
    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur, reference_data=ref)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(OUT_HTML))

    # Résumé lisible dans le terminal
    for metric in result.dict().get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount"):
            print(f"Colonnes en dérive : {value['count']:.0f} sur {len(cur.columns)} "
                  f"({value['share']:.0%})")
        elif name.startswith("ValueDrift") and isinstance(value, (int, float)):
            cfg = metric.get("config", {})
            method, threshold = cfg.get("method", ""), cfg.get("threshold", 0.05)
            # p-value : dérive si elle est PETITE ; distance : dérive si elle est GRANDE
            drift = value < threshold if "p_value" in method else value > threshold
            print(f"  {cfg.get('column', name):18} {'DÉRIVE' if drift else 'stable':7}"
                  f" ({method} = {value:.3f}, seuil {threshold})")
    print(f"✓ Rapport : {OUT_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
