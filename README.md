# NER Pipeline OSINT — Renseignement Militaire

> Extraction automatique d'entités militaires sur 21 742 articles TASS (2016–2026)  

---

## Objectif

Ce projet construit un pipeline NLP complet de bout en bout pour extraire des entités militaires depuis des articles de presse de l'agence TASS (média d'État russe). L'objectif est de détecter des signaux de renseignement — notamment une montée en tension militaire avant l'invasion de l'Ukraine en février 2022.

**3 labels d'entités extraites :**
- `MIL_UNIT` — Unités militaires (ex: *47th artillery brigade*, *VDV*)
- `MIL_ORG` — Organisations militaires (ex: *NATO*, *Russian Defense Ministry*, *Pentagon*)
- `MIL_WEAPON` — Systèmes d'armes (ex: *Kalibr*, *HIMARS*, *S-400*, *Geran-2*)

---

## Résultats clés

| Métrique | Valeur |
|---|---|
| Corpus total | 21 742 articles TASS (2016–2026) |
| Articles annotés (entraînement) | 500 articles via Mistral API |
| Entités dans le dataset d'entraînement | 6 023 (MIL_UNIT: 2421, MIL_ORG: 2300, MIL_WEAPON: 1302) |
| F-score du modèle NER | 0.444 (modèle vide, sans transfer learning) |
| Entités extraites sur le corpus complet | 255 615 |
| Arme la plus mentionnée | S-400 (564 articles) |

---

## Architecture du pipeline

```
data_set.json (21 742 articles)
        │
        ▼
[Étape 1] Échantillonnage aléatoire (seed=42)
        │
        ▼
articles_500.json (500 articles)
        │
        ▼
[Étape 2] 1_annotate_llm.py — Annotation via Mistral API
        │
        ▼
train_llm.json (387) + dev_llm.json (97)
        │
        ▼
[Étape 3] 2_finetune_ner.py — Fine-tuning spaCy NER
        │
        ▼
model_ner/model-best/ (modèle entraîné)
        │
        ▼
[Étape 4] 3_inference.py — Inférence sur les 21 742 articles
        │
        ▼
corpus_annotated.json (21 742 articles enrichis)
        │
        ▼
[Étape 5] 4_ingest_elasticsearch.py — Ingestion dans Elasticsearch
        │
        ▼
Index Elasticsearch → Dashboard Kibana
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Annotation LLM | Mistral API (`mistral-small-latest`) |
| Modèle NER | spaCy 3.8 |
| Base de données | Elasticsearch 9.4.3 |
| Visualisation | Kibana 9.4.3 |
| Langage | Python 3.13 |

---

## Installation

```bash
pip install spacy requests elasticsearch
```

> ⚠️ `en_core_web_sm` nécessite un accès à internet non filtré.  
> Si indisponible, le script `2_finetune_ner.py` utilise `spacy.blank("en")` automatiquement.

---

## Configuration — Variables à renseigner

Avant de lancer les scripts, remplace les valeurs suivantes :

**Dans `1_annotate_llm.py` :**
```python
API_KEY = "VOTRE_CLE_MISTRAL"   # Obtenir sur console.mistral.ai (gratuit)
```

**Dans `4_ingest_elasticsearch.py` :**
```python
ES_PASS = "VOTRE_MOT_DE_PASSE_ES"   # Défini lors du premier lancement d'Elasticsearch
```

---

## Lancer le pipeline (dans l'ordre)

### Étape 1 — Préparer les 500 articles

Assure-toi d'avoir `data_set.json` dans le dossier, puis extrais l'échantillon :

```python
import json, random
with open("data_set.json", encoding="utf-8") as f:
    data = json.load(f)
random.seed(42)
sample = random.sample([a for a in data if a.get("text") and len(a["text"]) > 100], 500)
with open("articles_500.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)
```

### Étape 2 — Annotation LLM

```bash
python 1_annotate_llm.py
```

Durée estimée : ~15 minutes pour 500 articles.  
Génère `train_llm.json` et `dev_llm.json`.

### Étape 3 — Fine-tuning NER

```bash
python 2_finetune_ner.py
```

Durée estimée : 20–40 minutes selon le CPU.  
Génère `model_ner/model-best/`.

### Étape 4 — Inférence sur le corpus complet

```bash
python 3_inference.py
```

Durée estimée : ~8 minutes pour 21 742 articles.  
Génère `corpus_annotated.json`.

### Étape 5 — Ingestion dans Elasticsearch

Lancer Elasticsearch 9.4.3 en local, puis :

```bash
python 4_ingest_elasticsearch.py
```

Durée estimée : ~2 minutes.  
Ouvrir Kibana sur `https://localhost:5601` pour les dashboards.

---

## Fichiers du repo

| Fichier | Description |
|---|---|
| `1_annotate_llm.py` | Annotation LLM via Mistral API |
| `2_finetune_ner.py` | Fine-tuning du modèle NER spaCy |
| `3_inference.py` | Inférence sur le corpus complet |
| `4_ingest_elasticsearch.py` | Ingestion dans Elasticsearch |
| `train_llm.json` | Dataset d'entraînement (387 articles annotés) |
| `dev_llm.json` | Dataset de validation (97 articles annotés) |

> ℹ️ `data_set.json` et `corpus_annotated.json` ne sont pas inclus dans ce repo (fichiers trop volumineux — plusieurs centaines de Mo).

---

## Problèmes rencontrés et solutions

| Problème | Solution |
|---|---|
| API Gemini — quota 0 | Switch vers Mistral API |
| `google.generativeai` déprécié | Migration vers `from google import genai` |
| `en_core_web_sm` indisponible | `spacy.blank("en")` utilisé à la place |
| Elasticsearch 9.x — HTTPS obligatoire | `verify_certs=False` + `basic_auth` |
| Kibana — enrollment token requis | `bin\elasticsearch-create-enrollment-token -s kibana` |


