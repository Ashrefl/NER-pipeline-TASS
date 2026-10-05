"""Tests de l'étape 3 — préparation des données et entraînement du modèle NER."""
import json


def _write(tmp_path, items):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return str(path)


def test_build_model_contient_les_trois_labels(finetune):
    nlp = finetune.build_model()
    assert set(nlp.get_pipe("ner").labels) == {"MIL_UNIT", "MIL_ORG", "MIL_WEAPON"}


def test_json_to_docs_convertit_les_entites(finetune, tmp_path):
    nlp = finetune.build_model()
    path = _write(tmp_path, [{"text": "NATO deploys HIMARS", "entities": [[0, 4, "MIL_ORG"], [13, 19, "MIL_WEAPON"]]}])
    docs = finetune.json_to_docs(path, nlp)
    assert [(e.text, e.label_) for e in docs[0].ents] == [("NATO", "MIL_ORG"), ("HIMARS", "MIL_WEAPON")]


def test_json_to_docs_ignore_les_articles_sans_entite(finetune, tmp_path):
    nlp = finetune.build_model()
    path = _write(tmp_path, [{"text": "Nothing here", "entities": []}])
    assert finetune.json_to_docs(path, nlp) == []


def test_entrainement_court_sauvegarde_un_modele(finetune, tmp_path):
    """Entraînement de 2 epochs sur un mini-jeu : le modèle doit être sauvegardé."""
    items = [{"text": f"NATO deploys HIMARS number {i}", "entities": [[0, 4, "MIL_ORG"], [13, 19, "MIL_WEAPON"]]} for i in range(8)]
    nlp = finetune.build_model()
    docs = finetune.json_to_docs(_write(tmp_path, items), nlp)
    examples = finetune.docs_to_examples(nlp, docs)
    best_f = finetune.train(nlp, examples, list(examples), model_out=str(tmp_path / "model"), epochs=2)
    assert 0.0 <= best_f <= 1.0
    assert (tmp_path / "model" / "model-best").exists()
