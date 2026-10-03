"""Tests de l'étape 4 — application du modèle NER au corpus."""
import spacy


def test_epoch_to_iso_conversion(inference):
    assert inference.epoch_to_iso(0) == "1970-01-01T00:00:00Z"
    assert inference.epoch_to_iso(1759071600) == "2025-09-28T15:00:00Z"


def test_epoch_to_iso_valeurs_invalides(inference):
    assert inference.epoch_to_iso(None) is None
    assert inference.epoch_to_iso("pas une date") is None


def test_build_text_concatene_titre_et_corps(inference):
    assert inference.build_text({"title": "Titre", "text": "Corps"}) == "Titre Corps"
    assert inference.build_text({"title": None, "text": "Corps"}) == "Corps"


def test_count_by_label(inference):
    results = [
        {"entities": [{"label": "MIL_ORG"}, {"label": "MIL_WEAPON"}]},
        {"entities": [{"label": "MIL_WEAPON"}]},
    ]
    assert inference.count_by_label(results) == {"MIL_UNIT": 0, "MIL_ORG": 1, "MIL_WEAPON": 2}


def test_annotate_corpus_bout_en_bout(inference):
    """Test d'intégration : un mini-modèle à règles remplace le modèle entraîné."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "MIL_WEAPON", "pattern": "HIMARS"}])

    articles = [{"id": 1, "title": "HIMARS", "text": "strike reported", "date": 0, "tags": ["UAV"]}]
    result = inference.annotate_corpus(nlp, articles)[0]

    assert result["id"] == 1
    assert result["date_iso"] == "1970-01-01T00:00:00Z"
    assert result["entities"] == [{"text": "HIMARS", "label": "MIL_WEAPON", "start": 0, "end": 6}]
