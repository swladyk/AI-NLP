# Oficjalny, lekki obraz Pythona 3.11
FROM python:3.11-slim

# Zmienne środowiskowe dla czystszego logowania i braku buforowania
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Zależności systemowe (git przydaje się przy instalacji niektórych paczek,
# build-essential bywa wymagany do kompilacji zależności natywnych)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Najpierw kopiujemy tylko requirements.txt, aby wykorzystać cache warstw Dockera
COPY requirements.txt /tmp/requirements.txt

# Aktualizacja pip i instalacja zależności projektu
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Pobranie modelu spaCy dla języka angielskiego (używany do przetwarzania tekstu)
RUN python -m spacy download en_core_web_sm

# Domyślna komenda — kontener utrzymywany przy życiu przez Dev Containers
CMD ["sleep", "infinity"]
