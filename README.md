# NER Pipeline OSINT — Renseignement militaire

Extraction automatique d'entités militaires dans **21 742 articles** de l'agence TASS (rubrique *Military & Defense*, 2016-2026).

Auteur : **Achraf LIMEM**

---

## Objectif

Transformer des dépêches en texte libre en données structurées et interrogeables. Un modèle de reconnaissance d'entités nommées (NER) détecte trois types d'entités :

| Label | Définition | Exemples |
|---|---|---|
| `MIL_UNIT` | Unités militaires | *47th artillery brigade*, *VDV* |
| `MIL_ORG` | Organisations, ministères, alliances | *NATO*, *Pentagon*, *Russian Defense Ministry* |
| `MIL_WEAPON` | Systèmes d'armes | *S-400*, *HIMARS*, *Kalibr* |

Aucun label de type personne n'est extrait (minimisation des données personnelles dès la conception).

---

## Résultats clés

| Indicateur | Valeur |
|---|---|
| Corpus | 21 742 articles |
| Articles annotés par LLM (Mistral) | 500, soit 6 023 entités |
| F-score global du modèle (jeu de validation) | 0,444 |
| Entités extraites sur le corpus complet | 255 615 |
| Arme la plus citée | S-400 (617 articles) |
| Documents indexés dans Elasticsearch | 21 742, 0 erreur |

---

## Architecture du pipeline

```
tass.com
   │  0_scrape_tass.py          collecte (RSS + pages HTML)
   ▼
data_set.json                   21 742 articles
   │  échantillonnage (seed 42)
   ▼
articles_500.json               500 articles
   │  1_annotate_llm.py         annotation via l'API Mistral
   ▼
train_llm.json / dev_llm.json   données d'entraînement (80 / 20)
   │  2_finetune_ner.py         entraînement du modèle spaCy
   ▼
model_ner/model-best/
   │  3_inference.py            application du modèle au corpus
   ▼
corpus_annotated.json           21 742 articles enrichis
   │  4_ingest_elasticsearch.py ingestion (API Bulk, HTTPS)
   ▼
Elasticsearch (index military_ner)  →  Kibana
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.13 |
| Collecte | requests, BeautifulSoup, lxml |
| Annotation | API Mistral (`mistral-small-latest`) |
| Modèle NER | spaCy 3.8 |
| Stockage et recherche | Elasticsearch 9.4.3 |
| Visualisation | Kibana 9.4.3 |
| Tests et intégration continue | pytest, GitHub Actions |

---

## Structure du dépôt

| Fichier / dossier | Rôle |
|---|---|
| `0_scrape_tass.py` | Collecte des articles TASS |
| `1_annotate_llm.py` | Annotation automatique via Mistral |
| `2_finetune_ner.py` | Entraînement du modèle NER |
| `3_inference.py` | Application du modèle aux 21 742 articles |
| `4_ingest_elasticsearch.py` | Indexation dans Elasticsearch |
| `train_llm.json`, `dev_llm.json` | Jeux d'entraînement et de validation annotés |
| `tests/` | Tests automatiques (pytest) |
| `.github/workflows/tests.yml` | Intégration continue : tests lancés à chaque push |
| `requirements.txt` | Dépendances Python |
| `.env.example` | Modèle de configuration des secrets |

Les fichiers volumineux (`data_set.json`, `corpus_annotated.json`, `model_ner/`) et le fichier `.env` ne sont pas versionnés (voir `.gitignore`).

---

## Installation

```bash
pip install -r requirements.txt
```

### Configuration des secrets

Les identifiants ne sont jamais écrits dans le code. Copier `.env.example` sous le nom `.env`, puis renseigner :

```
MISTRAL_API_KEY=...        # clé API Mistral (console.mistral.ai)
ES_HOST=https://localhost:9200
ES_USER=elastic
ES_PASS=...                # mot de passe Elasticsearch
```

Le fichier `.env` est exclu du dépôt par `.gitignore`.

---

## Exécution (dans l'ordre)

```bash
python 0_scrape_tass.py            # 1. collecte        → data_set.json
# 2. échantillonnage               → articles_500.json (voir ci-dessous)
python 1_annotate_llm.py           # 3. annotation      → train_llm.json, dev_llm.json
python 2_finetune_ner.py           # 4. entraînement    → model_ner/model-best/
python 3_inference.py              # 5. inférence       → corpus_annotated.json
python 4_ingest_elasticsearch.py   # 6. indexation      → index military_ner
```

Échantillonnage des 500 articles :

```python
import json, random
with open("data_set.json", encoding="utf-8") as f:
    data = json.load(f)
random.seed(42)
sample = random.sample([a for a in data if a.get("text") and len(a["text"]) > 100], 500)
with open("articles_500.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)
```

Elasticsearch doit être démarré avant l'indexation. Kibana est ensuite accessible sur `http://localhost:5601`.

---

## Tests

```bash
pytest -v
```

32 tests couvrent les fonctions critiques : extraction des champs du scraper (titre, texte, date, tags), nettoyage des réponses du LLM, calcul des positions des entités (hallucinations et chevauchements), conversion des dates, construction des documents Elasticsearch, ainsi qu'un test d'intégration de l'inférence.

Les tests sont lancés automatiquement par **GitHub Actions** à chaque push (onglet *Actions* du dépôt).

`2_finetune_ner.py` (entraînement complet, plusieurs dizaines de minutes) n'est pas couvert par les tests automatiques.

---

## Sécurité et données personnelles

- Secrets lus depuis un fichier `.env` non versionné.
- Échanges avec Elasticsearch en HTTPS avec authentification.
- Aucune entité de type personne extraite ; ni auteur ni commentaire collectés.
- Le détail figure dans les plans d'infrastructure et de pipeline.

---

## Limites et pistes d'amélioration (non réalisées)

| Limite | Piste |
|---|---|
| Étapes lancées manuellement | Planification (Planificateur de tâches / cron) et alertes par e-mail en cas d'erreur |
| Pas de réentraînement automatique | Réentraînement périodique et suivi de la dérive du modèle |
| Entités non normalisées (`UAV` / `UAVs`) | Normalisation des noms avant indexation |
| F-score de `MIL_UNIT` faible (0,355) | Modèle pré-entraîné, relecture humaine d'un échantillon d'annotations |
| Elasticsearch mono-nœud | Cluster de 3 nœuds avec réplication et snapshots |

---

## Problèmes rencontrés et solutions

| Problème | Solution |
|---|---|
| Quota nul sur l'API Gemini | Passage à l'API Mistral |
| `en_core_web_sm` indisponible | Modèle `spacy.blank("en")` |
| Elasticsearch 9 : HTTPS obligatoire | `basic_auth` + certificat auto-signé |
| Kibana : jeton d'enrôlement requis | `elasticsearch-create-enrollment-token -s kibana` |
| Comptage incomplet juste après l'ingestion | `es.indices.refresh()` avant le comptage |
