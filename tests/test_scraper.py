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


def test_extract_title_prefere_og_title(scraper):
    html = '<meta property="og:title" content="Titre officiel"><h1>Autre titre</h1>'
    assert scraper.extract_title(BeautifulSoup(html, "lxml")) == "Titre officiel"


def test_extract_title_repli_sur_h1(scraper):
    assert scraper.extract_title(BeautifulSoup("<h1>Titre H1</h1>", "lxml")) == "Titre H1"


def test_extract_text_repere_le_marqueur_tass(scraper):
    html = (
        "<nav><p>Menu de navigation du site avec plusieurs liens inutiles</p></nav>"
        "<div class='body'><p>MOSCOW, October 4. /TASS/. The Russian Defense Ministry said it strikes.</p>"
        "<p>Second paragraph of the article with enough characters to be kept.</p></div>"
    )
    text = scraper.extract_text(BeautifulSoup(html, "lxml"))
    assert text.startswith("MOSCOW, October 4. /TASS/.")
    assert "Second paragraph" in text
    assert "Menu de navigation" not in text


def test_extract_text_bloc_sans_paragraphe(scraper):
    corps = "MOSCOW, October 4. /TASS/. " + "Long article body without paragraph tags. " * 6
    html = f"<div class='news'>{corps}</div>"
    assert "/TASS/" in scraper.extract_text(BeautifulSoup(html, "lxml"))


def test_extract_date_ignore_la_date_du_pied_de_page(scraper):
    html = ("<p>Report published on September 28, 2025.</p>"
            "<footer>Certificate was issued on April 2, 1999</footer>")
    soup = BeautifulSoup(html, "lxml")
    assert scraper.extract_date(soup) == int(datetime(2025, 9, 28).timestamp())
