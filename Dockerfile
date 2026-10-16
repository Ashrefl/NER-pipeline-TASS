# Image de l'API de prédiction NER
# Construire :  docker build -t ner-tass-api .
# Lancer     :  docker run -p 8000:8000 ner-tass-api
FROM python:3.12-slim

WORKDIR /app

# Dépendances de l'API seulement : image plus légère que le projet complet
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Code de l'API et modèle entraîné
COPY api/ api/
COPY model_ner/model-best/ model_ner/model-best/

ENV MODEL_PATH=model_ner/model-best
EXPOSE 8000

# Vérification automatique de l'état du conteneur
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
