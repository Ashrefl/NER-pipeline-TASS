"""
ÉTAPE 2 — Tirage de l'échantillon de 500 articles
Entrée  : data_set.json       (le corpus complet, 21 742 articles)
Sortie  : articles_500.json   (500 articles tirés au hasard, à annoter)

Exécuter :
    python 0b_echantillon_500.py
"""

import json
import random

CORPUS_IN  = "data_set.json"
OUTPUT     = "articles_500.json"
TAILLE     = 500    # nombre d'articles à tirer
MIN_CHARS  = 100    # on écarte les articles vides ou trop courts
SEED       = 42     # graine fixe : le même tirage à chaque exécution


def main():
    with open(CORPUS_IN, encoding="utf-8") as f:
        articles = json.load(f)

    # 1. Filtre qualité : seulement les articles avec un vrai texte
    candidats = [a for a in articles if len(a.get("text") or "") > MIN_CHARS]

    # 2. Tirage au hasard, sans doublon, toujours le même grâce à la graine
    random.seed(SEED)
    echantillon = random.sample(candidats, TAILLE)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(echantillon, f, ensure_ascii=False, indent=2)

    print(f"{len(articles)} articles dans le corpus")
    print(f"{len(candidats)} articles avec plus de {MIN_CHARS} caractères")
    print(f"{TAILLE} articles tirés au hasard -> {OUTPUT}")


if __name__ == "__main__":
    main()
