"""
ÉTAPE 5 — Ingestion dans Elasticsearch 9.x
Entrée  : corpus_annotated.json
Index ES: military_ner

Prérequis :
    pip install elasticsearch
Exécuter :
    python 4_ingest_elasticsearch.py
"""

import json
from elasticsearch import Elasticsearch, helpers

ES_HOST  = "https://localhost:9200"
ES_USER  = "elastic"
ES_PASS  = "VOTRE_MOT_DE_PASSE_ES"
INDEX    = "military_ner"
INPUT    = "corpus_annotated.json"
BATCH    = 500

# ─── Connexion (SSL désactivé pour certificat auto-signé local) ───────────────
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASS),
    verify_certs=False,
    ssl_show_warn=False
)

if not es.ping():
    raise ConnectionError(f"Impossible de joindre Elasticsearch sur {ES_HOST}")
print(f"✓ Connecté à Elasticsearch {es.info()['version']['number']}")

# ─── Mapping de l'index ───────────────────────────────────────────────────────
MAPPING = {
    "mappings": {
        "properties": {
            "id":           {"type": "long"},
            "title":        {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "text":         {"type": "text"},
            "date_iso":     {"type": "date"},
            "date_epoch":   {"type": "long"},
            "tags":         {"type": "keyword"},
            "mark":         {"type": "keyword"},
            "link":         {"type": "keyword"},
            "isFlash":      {"type": "boolean"},
            "entities": {
                "type": "nested",
                "properties": {
                    "text":  {"type": "keyword"},
                    "label": {"type": "keyword"},
                }
            },
            "mil_units":    {"type": "keyword"},
            "mil_orgs":     {"type": "keyword"},
            "mil_weapons":  {"type": "keyword"},
            "entity_count": {"type": "integer"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0}
}

if es.indices.exists(index=INDEX):
    es.indices.delete(index=INDEX)
    print(f"Index '{INDEX}' supprimé")
es.indices.create(index=INDEX, body=MAPPING)
print(f"Index '{INDEX}' créé")

# ─── Chargement ──────────────────────────────────────────────────────────────
print(f"Chargement de {INPUT}...")
with open(INPUT, encoding="utf-8") as f:
    corpus = json.load(f)
print(f"{len(corpus)} articles chargés")

# ─── Générateur de documents ──────────────────────────────────────────────────
def generate_docs(corpus):
    for art in corpus:
        entities = art.get("entities", [])
        mil_units   = list({e["text"] for e in entities if e["label"] == "MIL_UNIT"})
        mil_orgs    = list({e["text"] for e in entities if e["label"] == "MIL_ORG"})
        mil_weapons = list({e["text"] for e in entities if e["label"] == "MIL_WEAPON"})

        yield {
            "_index": INDEX,
            "_id":    art["id"],
            "_source": {
                "id":           art["id"],
                "title":        art.get("title", ""),
                "text":         art.get("text", ""),
                "date_iso":     art.get("date_iso"),
                "date_epoch":   art.get("date_epoch"),
                "tags":         art.get("tags", []),
                "mark":         art.get("mark"),
                "link":         art.get("link", ""),
                "isFlash":      bool(art.get("isFlash", 0)),
                "entities":     entities,
                "mil_units":    mil_units,
                "mil_orgs":     mil_orgs,
                "mil_weapons":  mil_weapons,
                "entity_count": len(entities),
            }
        }

# ─── Ingestion ────────────────────────────────────────────────────────────────
print("Ingestion en cours...")
success, errors = helpers.bulk(
    es, generate_docs(corpus),
    chunk_size=BATCH,
    raise_on_error=False,
    stats_only=False,
)

print(f"\n=== INGESTION TERMINÉE ===")
print(f"Succès  : {success}")
print(f"Erreurs : {len(errors) if isinstance(errors, list) else errors}")
print(f"Documents dans '{INDEX}' : {es.count(index=INDEX)['count']}")

# ─── Test rapide ─────────────────────────────────────────────────────────────
print("\n--- Top 5 armes les plus mentionnées ---")
r = es.search(index=INDEX, body={
    "size": 0,
    "aggs": {"top_weapons": {"terms": {"field": "mil_weapons", "size": 5}}}
})
for b in r["aggregations"]["top_weapons"]["buckets"]:
    print(f"  {b['key']:30} {b['doc_count']:5d} articles")

print(f"\n✓ Prêt pour Kibana sur https://localhost:5601")