"""Tests de l'étape 0 — extraction des champs d'un article TASS."""
from datetime import datetime, timezone

from bs4 import BeautifulSoup


def test_extract_id_depuis_url(scraper):
    assert scraper.extract_id("https://tass.com/defense/2194041") == 2194041


def test_extract_id_url_sans_identifiant(scraper):
    assert scraper.extract_id("https://tass.com/defense/") == 0


def test_extract_date_balise_time(scraper):
    soup = BeautifulSoup('<time datetime="2025-09-28T15:00:00Z"></time>', "lxml")
    attendu = int(datetime(2025, 9, 28, 15, 0, tzinfo=timezone.utc).timestamp())
    assert scraper.extract_date(soup) == attendu


def test_extract_date_meta_published_time(scraper):
    html = '<meta property="article:published_time" content="2024-02-24T06:00:00Z">'
    soup = BeautifulSoup(html, "lxml")
    attendu = int(datetime(2024, 2, 24, 6, 0, tzinfo=timezone.utc).timestamp())
    assert scraper.extract_date(soup) == attendu


def test_extract_date_texte_en_dernier_recours(scraper):
    soup = BeautifulSoup("<p>MOSCOW, September 28, 2025. Report.</p>", "lxml")
    assert scraper.extract_date(soup) == int(datetime(2025, 9, 28).timestamp())


def test_extract_date_introuvable_renvoie_zero(scraper):
    soup = BeautifulSoup("<p>Aucune date ici</p>", "lxml")
    assert scraper.extract_date(soup) == 0


def test_extract_tags_sans_doublon(scraper):
    html = '<a class="tag">UAV</a><a class="tags-item">UAV</a><a class="tag">NATO</a>'
    soup = BeautifulSoup(html, "lxml")
    assert scraper.extract_tags(soup) == ["UAV", "NATO"]
