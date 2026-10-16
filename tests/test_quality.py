"""Tests du contrôle qualité des données collectées."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_quality", ROOT / "scripts" / "check_quality.py")
cq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cq)

GOOD = {"id": 1, "title": "t", "text": "word " * 40, "date": 1700000000, "link": "x"}


def test_donnees_propres():
    r = cq.check([GOOD, {**GOOD, "id": 2}])
    assert r["blocking"] is False and r["duplicate_ids"] == 0


def test_doublon_bloquant():
    assert cq.check([GOOD, GOOD])["blocking"] is True


def test_champ_manquant_bloquant():
    assert cq.check([{"id": 1, "title": "t"}])["blocking"] is True


def test_texte_court_et_date_future_signales():
    r = cq.check([{**GOOD, "text": "trop court", "date": 4102444800}], now=1700000000)
    assert r["empty_or_short_text"] == 1 and r["future_date"] == 1 and r["blocking"] is False
