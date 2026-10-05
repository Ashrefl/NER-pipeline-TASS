"""
Configuration commune des tests.

Les scripts du projet commencent par un chiffre (0_scrape_tass.py, ...),
ce qui empêche un « import » Python classique : on les charge par leur chemin.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Valeurs factices : les tests n'appellent jamais l'API Mistral ni Elasticsearch
os.environ.setdefault("MISTRAL_API_KEY", "cle-de-test")
os.environ.setdefault("ES_PASS", "mot-de-passe-de-test")


def load_script(filename):
    """Charge un script du projet comme un module, sans exécuter son main()."""
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def scraper():
    return load_script("0_scrape_tass.py")


@pytest.fixture(scope="session")
def annotation():
    return load_script("1_annotate_llm.py")


@pytest.fixture(scope="session")
def inference():
    return load_script("3_inference.py")


@pytest.fixture(scope="session")
def ingestion():
    return load_script("4_ingest_elasticsearch.py")


@pytest.fixture(scope="session")
def finetune():
    return load_script("2_finetune_ner.py")
