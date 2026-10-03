"""Tests de l'étape 2 — nettoyage des réponses du LLM et calcul des positions."""


# ─── parse_response ───────────────────────────────────────────────────────────
def test_parse_response_retire_les_balises_markdown(annotation):
    raw = '```json\n[[{"text": "NATO", "label": "MIL_ORG"}], []]\n```'
    assert annotation.parse_response(raw, 2) == [[{"text": "NATO", "label": "MIL_ORG"}], []]


def test_parse_response_complete_les_listes_manquantes(annotation):
    raw = '[[{"text": "NATO", "label": "MIL_ORG"}]]'
    result = annotation.parse_response(raw, 3)
    assert len(result) == 3
    assert result[1] == [] and result[2] == []


def test_parse_response_tronque_les_listes_en_trop(annotation):
    raw = "[[], [], [], []]"
    assert len(annotation.parse_response(raw, 2)) == 2


def test_parse_response_liste_plate_attribuee_au_premier_article(annotation):
    raw = '[{"text": "HIMARS", "label": "MIL_WEAPON"}]'
    assert annotation.parse_response(raw, 2) == [[{"text": "HIMARS", "label": "MIL_WEAPON"}], []]


def test_parse_response_json_invalide_ne_plante_pas(annotation):
    assert annotation.parse_response("ceci n'est pas du JSON", 2) == [[], []]


def test_parse_response_objet_au_lieu_de_liste(annotation):
    assert annotation.parse_response('{"erreur": true}', 2) == [[], []]


# ─── find_spans ───────────────────────────────────────────────────────────────
def test_find_spans_positions_exactes(annotation):
    text = "NATO deploys HIMARS"
    spans = annotation.find_spans(text, [{"text": "HIMARS", "label": "MIL_WEAPON"}])
    assert spans == [{"start": 13, "end": 19, "label": "MIL_WEAPON"}]
    assert text[13:19] == "HIMARS"


def test_find_spans_toutes_les_occurrences(annotation):
    spans = annotation.find_spans("NATO and NATO allies", [{"text": "NATO", "label": "MIL_ORG"}])
    assert [(s["start"], s["end"]) for s in spans] == [(0, 4), (9, 13)]


def test_find_spans_rejette_un_label_inconnu(annotation):
    assert annotation.find_spans("Vladimir visited", [{"text": "Vladimir", "label": "PERSON"}]) == []


def test_find_spans_ignore_une_entite_absente_du_texte(annotation):
    # Protection contre les hallucinations du LLM
    assert annotation.find_spans("NATO summit", [{"text": "Kalibr", "label": "MIL_WEAPON"}]) == []


def test_find_spans_supprime_les_chevauchements(annotation):
    text = "S-400 Triumf system"
    ents = [{"text": "S-400 Triumf", "label": "MIL_WEAPON"}, {"text": "Triumf", "label": "MIL_WEAPON"}]
    assert annotation.find_spans(text, ents) == [{"start": 0, "end": 12, "label": "MIL_WEAPON"}]
