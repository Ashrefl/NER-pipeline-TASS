"""
ÉTAPE 0 — Scraping TASS Military & Defense
Sortie  : data_set.json (même format que le corpus original)

Stratégie :
  1. Récupère les IDs d'articles via le flux RSS TASS
  2. Scrape chaque article individuellement (HTML server-side)
  3. Sauvegarde progressive toutes les 50 articles (reprise possible)

Prérequis :
    pip install requests beautifulsoup4 lxml
Exécuter :
    python 0_scrape_tass.py
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

# ─── Configuration ────────────────────────────────────────────────────────────
RSS_URL     = "https://tass.com/rss/v2.xml"
BASE_URL    = "https://tass.com"
SECTION     = "/defense"
OUTPUT      = "data_set.json"
DELAY       = 1.5          # secondes entre chaque requête (respectueux)
MAX_RETRIES = 3            # tentatives par article avant abandon
SAVE_EVERY  = 50           # sauvegarde intermédiaire toutes les N articles
MAX_PAGES   = 10           # nombre de pages RSS à parcourir

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# Dates de publication lues dans le flux RSS (URL → EPOCH), utilisées en repli
RSS_DATES = {}


# ─── Collecte des URLs via RSS ────────────────────────────────────────────────
def get_article_urls_from_rss():
    """Récupère les URLs des articles défense depuis le flux RSS TASS."""
    log.info(f"Récupération des articles via RSS : {RSS_URL}")
    urls = []
    try:
        r = requests.get(RSS_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            link = item.findtext("link")
            if link and SECTION in link:
                urls.append(link.strip())
                pub = item.findtext("pubDate")
                if pub:
                    try:
                        RSS_DATES[link.strip()] = int(parsedate_to_datetime(pub).timestamp())
                    except (TypeError, ValueError):
                        pass
        log.info(f"  {len(urls)} URLs défense trouvées dans le RSS")
    except Exception as e:
        log.warning(f"  Erreur RSS : {e}")
    return urls


def get_article_urls_by_id(start_id: int, end_id: int, step: int = 1):
    """
    Génère des URLs par itération d'IDs séquentiels.
    Utile pour scraper des périodes historiques (2016-2022).

    Exemple d'usage :
        urls = get_article_urls_by_id(800000, 900000)
    """
    return [f"{BASE_URL}{SECTION}/{i}" for i in range(start_id, end_id, step)]


# ─── Extraction d'un article ──────────────────────────────────────────────────
def parse_article(url: str, session: requests.Session) -> dict | None:
    """
    Scrape un article TASS et retourne un dict au format data_set.json.
    Retourne None si l'article est inaccessible ou hors-section.
    """
    article_id = extract_id(url)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)

            # Article supprimé ou redirigé hors-section
            if r.status_code == 404:
                log.warning(f"  404 — article {article_id} inexistant")
                return None
            if r.status_code != 200:
                log.warning(f"  HTTP {r.status_code} pour {url} (tentative {attempt})")
                time.sleep(DELAY * 2)
                continue

            soup = BeautifulSoup(r.text, "lxml")

            # ── Titre ──
            title = extract_title(soup)
            if not title:
                log.warning(f"  Titre introuvable pour {url}")
                return None

            # ── Texte ──
            text = extract_text(soup)
            if not text:
                log.warning(f"  Texte introuvable pour {url}")

            # ── Date ──
            # 1) balise <time> ou meta og:published_time
            # priorité à la date du flux RSS (fiable), sinon extraction depuis la page
            date_epoch = RSS_DATES.get(url) or extract_date(soup)

            # ── Tags ──
            tags = extract_tags(soup)

            # ── Mark (catégorie éditoriale) ──
            mark = extract_mark(soup)

            # ── isFlash ──
            is_flash = "flash" in url.lower() or "flash" in title.lower()

            return {
                "id":      article_id,
                "date":    date_epoch,
                "title":   title,
                "text":    text,
                "tags":    tags,
                "mark":    mark,
                "isFlash": int(is_flash),
                "link":    f"{SECTION}/{article_id}",
            }

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout pour {url} (tentative {attempt}/{MAX_RETRIES})")
            time.sleep(DELAY * 3)
        except Exception as e:
            log.warning(f"  Erreur {url} : {e} (tentative {attempt}/{MAX_RETRIES})")
            time.sleep(DELAY * 2)

    log.error(f"  Abandon après {MAX_RETRIES} tentatives : {url}")
    return None


def extract_title(soup: BeautifulSoup) -> str:
    """Extrait le titre : balise meta og:title, sinon premier <h1>."""
    meta = soup.find("meta", property="og:title")
    if meta and meta.get("content"):
        return meta["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def extract_text(soup: BeautifulSoup) -> str:
    """
    Extrait le corps de l'article.
    Méthode 1 : repère la mention « /TASS/ » qui ouvre chaque dépêche
                et récupère les paragraphes du bloc qui la contient.
    Méthode 2 : paragraphes de la balise <article> ou d'un bloc « text ».
    Méthode 3 : tous les paragraphes de la page.
    """
    def paragraphs_of(block):
        return [p.get_text(" ", strip=True) for p in block.find_all("p")
                if len(p.get_text(strip=True)) > 30]

    # Méthode 1
    marker = soup.find(string=re.compile(r"/TASS/"))
    if marker:
        block = marker.find_parent(["article", "section", "div"])
        while block is not None:
            paras = paragraphs_of(block)
            if paras:
                return " ".join(paras)
            # bloc sans <p> : on prend son texte brut s'il est assez long
            raw = block.get_text(" ", strip=True)
            if len(raw) > 200:
                return raw
            block = block.find_parent(["article", "section", "div"])

    # Méthode 2
    body = soup.find("article") or soup.find("div", class_=re.compile(r"text|body|content", re.I))
    if body:
        paras = paragraphs_of(body)
        if paras:
            return " ".join(paras)

    # Méthode 3
    return " ".join(paragraphs_of(soup))


def extract_id(url: str) -> int:
    """Extrait l'ID numérique de l'URL."""
    match = re.search(r"/(\d+)$", url)
    return int(match.group(1)) if match else 0


def extract_date(soup: BeautifulSoup) -> int:
    """Extrait la date de publication en timestamp EPOCH."""
    # Méthode 1 : balise <time datetime="...">
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        try:
            dt = datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            pass

    # Méthode 2 : meta og:published_time
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        try:
            dt = datetime.fromisoformat(meta["content"].replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            pass

    # Méthode 3 : dates « Month DD, YYYY » dans le texte de la page.
    # On garde la plus récente : le pied de page contient une date ancienne
    # (certificat d'enregistrement de 1999) qui ne doit pas être retenue.
    dates = []
    for match in re.finditer(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),?\s+(\d{4})",
        soup.get_text()
    ):
        try:
            dt = datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", "%B %d %Y")
            if dt <= datetime.now():
                dates.append(dt)
        except ValueError:
            pass
    if dates:
        return int(max(dates).timestamp())

    return 0  # date inconnue


def extract_tags(soup: BeautifulSoup) -> list[str]:
    """Extrait les tags/mots-clés de l'article."""
    tags = []

    # Balises <a> avec classe contenant "tag"
    for a in soup.find_all("a", class_=re.compile(r"tag", re.I)):
        t = a.get_text(strip=True)
        if t and len(t) < 50:
            tags.append(t)

    # Meta keywords
    if not tags:
        meta = soup.find("meta", attrs={"name": "keywords"})
        if meta and meta.get("content"):
            tags = [k.strip() for k in meta["content"].split(",") if k.strip()]

    return list(dict.fromkeys(tags))  # dédupliqué, ordre préservé


def extract_mark(soup: BeautifulSoup) -> str:
    """Extrait la catégorie éditoriale de l'article."""
    # Balise breadcrumb ou label de catégorie
    for selector in ["span.mark", "div.mark", "span.category", ".article-header__category"]:
        el = soup.select_one(selector)
        if el:
            return el.get_text(strip=True)

    # Meta og:section
    meta = soup.find("meta", property="article:section")
    if meta and meta.get("content"):
        return meta["content"]

    return "Military & Defense"


# ─── Pipeline principal ───────────────────────────────────────────────────────
def main():
    log.info("=== SCRAPER TASS — Military & Defense ===")

    # Reprise si fichier existant
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            articles = json.load(f)
        done_ids = {a["id"] for a in articles}
        log.info(f"Reprise : {len(articles)} articles déjà scrapés")
    else:
        articles = []
        done_ids = set()

    # Collecte des URLs
    urls = get_article_urls_from_rss()

    # Optionnel : ajouter des URLs par plage d'IDs pour les données historiques
    # urls += get_article_urls_by_id(start_id=800000, end_id=2200000, step=50)
    # urls = list(dict.fromkeys(urls))  # dédupliqué

    urls = [u for u in urls if extract_id(u) not in done_ids]
    log.info(f"{len(urls)} articles à scraper")

    session = requests.Session()
    errors = 0

    for i, url in enumerate(urls, 1):
        log.info(f"[{i}/{len(urls)}] {url}")

        article = parse_article(url, session)

        if article and article["text"]:
            articles.append(article)
            done_ids.add(article["id"])
            log.info(f"  ✓ '{article['title'][:60]}...' ({len(article['text'])} chars)")
        else:
            errors += 1
            log.warning(f"  ✗ Article ignoré")

        # Sauvegarde intermédiaire
        if i % SAVE_EVERY == 0:
            save(articles)
            log.info(f"  → Sauvegarde intermédiaire : {len(articles)} articles")

        time.sleep(DELAY)

    # Sauvegarde finale
    save(articles)

    log.info("\n=== SCRAPING TERMINÉ ===")
    log.info(f"Articles récupérés : {len(articles)}")
    log.info(f"Erreurs/ignorés    : {errors}")
    log.info(f"Fichier de sortie  : {OUTPUT}")
    log.info("→ Prochaine étape  : lance 1_annotate_llm.py")


def save(articles: list):
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
