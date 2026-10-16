"""Tests de l'API de prédiction et de la règle de réentraînement."""
import importlib.util
from pathlib import Path

import pytest
import spacy
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """API branchée sur un mini-modèle à règles : pas besoin du vrai modèle."""
    from api import main
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "MIL_WEAPON", "pattern": "S-400"},
                        {"label": "MIL_ORG", "pattern": "NATO"}])
    monkeypatch.setattr(main, "_nlp", nlp)
    monkeypatch.setattr(main, "LOG_PATH", tmp_path / "log.jsonl")
    return TestClient(main.app), tmp_path / "log.jsonl"


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_trouve_les_entites(client):
    c, _ = client
    r = c.post("/predict", json={"text": "NATO is worried about the S-400."})
    assert r.status_code == 200
    labels = sorted(e["label"] for e in r.json()["entities"])
    assert labels == ["MIL_ORG", "MIL_WEAPON"]
    assert r.json()["counts"]["MIL_WEAPON"] == 1


def test_predict_refuse_un_texte_vide(client):
    c, _ = client
    assert c.post("/predict", json={"text": ""}).status_code == 422


def test_predict_ecrit_le_journal_de_production(client):
    c, log = client
    c.post("/predict", json={"text": "The S-400 arrived."})
    line = log.read_text(encoding="utf-8")
    assert '"n_mil_weapon": 1' in line and "S-400" not in line   # le texte n'est pas stocké


def test_predict_batch(client):
    c, _ = client
    r = c.post("/predict/batch", json={"texts": ["S-400", "nothing here"]})
    assert [len(x["entities"]) for x in r.json()["results"]] == [1, 0]


def test_regle_de_redeploiement():
    spec = importlib.util.spec_from_file_location("retrain", ROOT / "retrain" / "retrain.py")
    retrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrain)
    assert retrain.decide(0.444, 0.470) is True     # vrai progrès : on déploie
    assert retrain.decide(0.444, 0.446) is False    # gain trop faible : on garde
    assert retrain.decide(0.444, 0.400) is False    # moins bon : on garde
