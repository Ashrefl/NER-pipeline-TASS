"""Tests de l'étape 5 — construction des documents Elasticsearch."""

ARTICLE = {
    "id": 2194041,
    "title": "Strike",
    "text": "NATO and NATO see HIMARS",
    "date_iso": "2025-09-28T15:00:00Z",
    "date_epoch": 1759071600,
    "isFlash": 1,
    "entities": [
        {"text": "NATO", "label": "MIL_ORG"},
        {"text": "NATO", "label": "MIL_ORG"},
        {"text": "HIMARS", "label": "MIL_WEAPON"},
    ],
}


def test_build_document_utilise_l_id_tass(ingestion):
    doc = ingestion.build_document(ARTICLE)
    assert doc["_id"] == 2194041
    assert doc["_index"] == "military_ner"


def test_build_document_champs_denormalises_sans_doublon(ingestion):
    source = ingestion.build_document(ARTICLE)["_source"]
    assert source["mil_orgs"] == ["NATO"]
    assert source["mil_weapons"] == ["HIMARS"]
    assert source["mil_units"] == []


def test_build_document_comptage_et_types(ingestion):
    source = ingestion.build_document(ARTICLE)["_source"]
    assert source["entity_count"] == 3
    assert source["isFlash"] is True


def test_mapping_types_cles(ingestion):
    props = ingestion.MAPPING["mappings"]["properties"]
    assert props["date_iso"]["type"] == "date"
    assert props["mil_weapons"]["type"] == "keyword"
    assert props["entities"]["type"] == "nested"
