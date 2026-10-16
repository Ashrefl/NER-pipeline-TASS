"""
Contrôle qualité d'un fichier d'articles collectés (sortie de 0_scrape_tass.py).

    python scripts/check_quality.py data_set.json
    python scripts/check_quality.py data_set.json --report quality_report.json

Bloquant (code de sortie 1) : champs obligatoires manquants, doublons d'id.
Avertissement seulement : aucun article, texte vide ou très court, date absente ou future.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone

REQUIRED  = ("id", "title", "text", "date", "link")
MIN_WORDS = 30


def check(articles, now=None):
    now = now or datetime.now(timezone.utc).timestamp()
    ids = Counter(a.get("id") for a in articles)
    report = {
        "articles": len(articles),
        "missing_fields": sum(1 for a in articles if any(k not in a for k in REQUIRED)),
        "duplicate_ids": sum(c - 1 for c in ids.values() if c > 1),
        "empty_or_short_text": sum(1 for a in articles if len((a.get("text") or "").split()) < MIN_WORDS),
        "missing_date": sum(1 for a in articles if not a.get("date")),
        "future_date": sum(1 for a in articles if a.get("date") and int(a["date"]) > now + 86400),
    }
    report["blocking"] = report["missing_fields"] > 0 or report["duplicate_ids"] > 0
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--report")
    args = p.parse_args()

    try:
        with open(args.path, encoding="utf-8") as f:
            articles = json.load(f)
    except FileNotFoundError:
        articles = []
    report = check(articles)

    for k, v in report.items():
        print(f"{k:22} {v}")
    if report["articles"] == 0:
        print("::warning::Aucun nouvel article collecté")
    if report["empty_or_short_text"] or report["missing_date"] or report["future_date"]:
        print("::warning::Certains articles sont incomplets, voir le rapport")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    sys.exit(1 if report["blocking"] else 0)


if __name__ == "__main__":
    main()
