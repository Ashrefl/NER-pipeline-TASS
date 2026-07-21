"""
ÉTAPE 2 — Annotation LLM via Mistral API
Entrée  : articles_500.json
Sortie  : annotations_llm.json + train_llm.json + dev_llm.json

Prérequis : pip install requests  (généralement déjà installé)
Exécuter  : python 1_annotate_llm.py
"""

import json, time, re, random, os, requests

API_KEY = "VOTRE_CLE_MISTRAL"
MODEL   = "mistral-small-latest"
INPUT   = "articles_500.json"
OUTPUT  = "annotations_llm.json"
BATCH   = 3      # articles par requête
DELAY   = 2.0    # secondes entre requêtes

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# ─── Prompt ──────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a military Named Entity Recognition expert.
Extract named entities from the {n} military news articles below.

Labels:
- MIL_UNIT   : specific military units (e.g. "47th artillery brigade", "joint group of forces")
- MIL_ORG    : military organizations, ministries, alliances (e.g. "NATO", "Pentagon", "Russian Defense Ministry")
- MIL_WEAPON : weapons, missiles, drones, tanks, aircraft, systems (e.g. "Kalibr", "HIMARS", "Geran-2", "S-400")

Return ONLY a JSON array of arrays — one array per article:
[[{{"text":"...", "label":"..."}}, ...], [...], ...]
Use the EXACT text as it appears. If no entities in an article, use [].

{articles}"""

def call_mistral(prompt):
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=HEADERS,
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def parse_response(raw, n_articles):
    """Parse la réponse JSON de Mistral."""
    # Enlever les blocs markdown ```json ... ```
    raw = re.sub(r'^```json\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        parsed = json.loads(raw)
        # Tableau de tableaux (format attendu)
        if parsed and isinstance(parsed[0], list):
            while len(parsed) < n_articles:
                parsed.append([])
            return parsed[:n_articles]
        # Tableau plat → premier article seulement
        return [parsed] + [[] for _ in range(n_articles - 1)]
    except:
        return [[] for _ in range(n_articles)]

def find_spans(text, entities):
    """Convertit les entités textuelles en spans (start, end, label)."""
    spans, seen = [], set()
    for ent in entities:
        t = ent.get("text", "").strip()
        label = ent.get("label", "")
        if not t or label not in ("MIL_UNIT", "MIL_ORG", "MIL_WEAPON"):
            continue
        idx = 0
        while True:
            pos = text.find(t, idx)
            if pos == -1:
                break
            key = (pos, pos + len(t))
            if key not in seen:
                seen.add(key)
                spans.append({"start": pos, "end": pos + len(t), "label": label})
            idx = pos + 1
    spans.sort(key=lambda x: x["start"])
    clean, last = [], -1
    for s in spans:
        if s["start"] >= last:
            clean.append(s)
            last = s["end"]
    return clean

def annotate_batch(articles):
    articles_text = ""
    for i, art in enumerate(articles):
        full = (art.get("title") or "") + " " + (art.get("text") or "")
        articles_text += f"\n\n--- ARTICLE {i+1} ---\n{full.strip()[:2500]}"

    prompt = PROMPT_TEMPLATE.format(n=len(articles), articles=articles_text)
    raw = call_mistral(prompt)
    return parse_response(raw, len(articles))

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(INPUT, encoding="utf-8") as f:
        articles = json.load(f)

    # Reprendre si interrompu
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            done = json.load(f)
        done_ids = {a["id"] for a in done}
        articles = [a for a in articles if a["id"] not in done_ids]
        print(f"Reprise : {len(done)} déjà annotés, {len(articles)} restants")
    else:
        done = []

    total = len(articles)
    n_batches = (total + BATCH - 1) // BATCH
    counts = {"MIL_UNIT": 0, "MIL_ORG": 0, "MIL_WEAPON": 0}

    print(f"Annotation de {total} articles via Mistral ({MODEL})...")
    print(f"Temps estimé : ~{n_batches * DELAY / 60:.0f} min\n")

    for i in range(0, total, BATCH):
        batch = articles[i:i+BATCH]
        batch_num = i // BATCH + 1

        try:
            batch_ents = annotate_batch(batch)
        except Exception as e:
            print(f"  Erreur batch {batch_num}: {e} — retry 10s...")
            time.sleep(10)
            try:
                batch_ents = annotate_batch(batch)
            except:
                batch_ents = [[] for _ in batch]

        for art, raw_ents in zip(batch, batch_ents):
            full = ((art.get("title") or "") + " " + (art.get("text") or "")).strip()
            spans = find_spans(full, raw_ents)
            for s in spans:
                counts[s["label"]] += 1
            done.append({"id": art["id"], "title": art.get("title",""),
                         "date": art.get("date"), "tags": art.get("tags",[]),
                         "text": full, "entities": spans})

        # Sauvegarde toutes les 10 batches
        if batch_num % 10 == 0 or batch_num == n_batches:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=2)
            pct = len(done) / 500 * 100
            print(f"  [{batch_num}/{n_batches}] {len(done)}/500 ({pct:.0f}%) — {counts}")

        time.sleep(DELAY)

    # Sauvegarde finale
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Annotation terminée !")
    print(f"  MIL_ORG    : {counts['MIL_ORG']}")
    print(f"  MIL_WEAPON : {counts['MIL_WEAPON']}")
    print(f"  MIL_UNIT   : {counts['MIL_UNIT']}")

    # Format spaCy + split 80/20
    spacy_fmt = [
        {"text": a["text"], "entities": [[e["start"], e["end"], e["label"]] for e in a["entities"]]}
        for a in done if a["entities"]
    ]
    random.seed(42)
    random.shuffle(spacy_fmt)
    split = int(len(spacy_fmt) * 0.8)

    with open("train_llm.json", "w", encoding="utf-8") as f:
        json.dump(spacy_fmt[:split], f, ensure_ascii=False, indent=2)
    with open("dev_llm.json", "w", encoding="utf-8") as f:
        json.dump(spacy_fmt[split:], f, ensure_ascii=False, indent=2)

    print(f"\n  train_llm.json : {split} articles")
    print(f"  dev_llm.json   : {len(spacy_fmt)-split} articles")
    print("\n→ Prochaine étape : lance 2_finetune_ner.py")