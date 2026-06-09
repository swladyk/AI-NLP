# AI-NLP — System analizy recenzji (Summarization, Keyword Extraction & Sentiment Analysis)

Akademicki projekt NLP budujący pipeline, który automatycznie przetwarza recenzje
produktów ze zbioru **Amazon Fine Food Reviews** na trzy sposoby:

1. **Text Summarization** — streszczanie recenzji (model `google/flan-t5-xl`),
2. **Keyword Extraction** — wyłuskiwanie słów/fraz kluczowych (`KeyBERT`),
3. **Sentiment Analysis** — klasyfikacja wydźwięku pozytywny/negatywny (`VADER`).

---

## Podział pracy

| Część | Zakres | Autor | Status |
|-------|--------|-------|--------|
| **Część A** | Text Summarization + Keyword Extraction | **Stanisław** | ✅ **Gotowe** |
| **Część B** | Sentiment Analysis | **Partner (do zrobienia)** | ⬜ **Do dokończenia** |

> **WAŻNE:** Sekcja **6** notatnika dotycząca **text summarization** i **keyword
> extraction** (podsekcje **6.1 – 6.3**) jest **już zrobiona przeze mnie
> (Stanisław)**. Twoim zadaniem jest **dokończenie analizy sentymentu (Sentiment
> Analysis)** — czyli podsekcje **6.4 i 6.5** oraz powiązane miejsca oznaczone
> jako `TODO (Partner)`. Reszta notatnika (wstęp, opis bibliotek, wczytanie i
> przygotowanie danych) jest wspólna i nie trzeba jej ruszać.

### Co dokładnie masz uzupełnić (wszystkie oznaczone `TODO (Partner)`)

W pliku `Projekt_NLP.md` (lub `Projekt_NLP.ipynb`):

- **Sekcja 2.3 — Sentiment Analysis (Science & Practice):** opis przykładów
  naukowych i biznesowych analizy sentymentu (np. lexicon-based VADER, modele
  transformerowe; zastosowania: monitoring marki, analiza opinii klientów).
- **Sekcja 3.3 — VADER:** opis biblioteki `vaderSentiment`
  (`SentimentIntensityAnalyzer`, metoda `polarity_scores()` z kluczami
  `neg`/`neu`/`pos`/`compound`, próg `compound >= 0.05` dla klasy pozytywnej).
- **Sekcja 6.4 — implementacja VADER:** policzenie `polarity_scores()` dla każdej
  recenzji z `df_sample` i zmapowanie `compound` → etykieta
  (positive / negative / neutral). Następnie zbudowanie "ground truth" z kolumny
  `Score` (np. `Score >= 4` → positive, `Score <= 2` → negative).
- **Sekcja 6.5 — metryki ewaluacji:** Accuracy, F1, `classification_report`
  oraz `confusion_matrix` z wizualizacją (np. heatmapa seaborn).
- **Sekcja 7 — Strengths & Weaknesses:** dopisanie mocnych i słabych stron VADER.

Korzystaj z gotowego `df_sample` (przygotowanego w sekcji 5.2) — jest wspólny dla
obu części, więc oboje pracujemy na tym samym podzbiorze danych.

---

## Struktura repozytorium

| Plik | Opis |
|------|------|
| `Projekt_NLP.ipynb` | Główny notatnik Jupyter — tu pracujesz. |
| `Projekt_NLP.md` | Ten sam notatnik w formacie Markdown (jupytext) — wygodny do podglądu/diffów w gicie. |
| `requirements.txt` | Zamrożone wersje bibliotek (pinned) z działającego środowiska. |
| `summarization_results.csv` | Przykładowy wynik Części A (oryginał + streszczenie + słowa kluczowe). |
| `.devcontainer/` | Konfiguracja Dev Container (gotowe środowisko w Dockerze). |

> Notatnik `.ipynb` i `.md` są zsynchronizowane przez **jupytext**. Edytuj
> najlepiej `.ipynb` w Jupyterze; `.md` jest wersją tekstową do gita.

---

## Jak uruchomić

### Wymagania wstępne

- Python 3.11
- Zalecane: GPU z ~6 GB VRAM dla modelu Flan-T5 XL (Część A). Bez GPU
  streszczanie działa, ale jest wolne. **Część B (VADER) jest lekka i działa
  szybko na CPU.**
- Konto Kaggle (do pobrania zbioru danych przez `kagglehub`).

### Wariant 1 — Dev Container (najprościej)

Repozytorium zawiera gotową konfigurację `.devcontainer/`. Jeśli używasz VS Code
z rozszerzeniem **Dev Containers**:

1. Otwórz folder projektu w VS Code.
2. `F1` → **Dev Containers: Reopen in Container**.
3. Środowisko z wszystkimi zależnościami zbuduje się automatycznie.

> ⚠️ **Jeśli NIE masz GPU** — usuń (lub zakomentuj) flagę GPU w pliku
> `.devcontainer/devcontainer.json`, inaczej kontener **nie wystartuje**.
> Chodzi o ten fragment:
>
> ```json
>     "runArgs": [
>         "--gpus",
>         "all"
>     ],
> ```
>
> Usuń cały blok `"runArgs"` (albo zostaw `"runArgs": []`) i przebuduj kontener
> (`F1` → **Dev Containers: Rebuild Container**). Część B (VADER) i tak nie
> potrzebuje GPU, więc na CPU wszystko zadziała.

### Wariant 2 — lokalne środowisko wirtualne

```powershell
# Utwórz i aktywuj środowisko wirtualne
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom Jupyter
jupyter notebook Projekt_NLP.ipynb
```

### Pobranie danych

Zbiór **Amazon Fine Food Reviews** pobiera się automatycznie w sekcji 5.1
notatnika przez `kagglehub`. Przy pierwszym uruchomieniu może być potrzebne
zalogowanie do Kaggle (token API w `~/.kaggle/kaggle.json`). Zob.
[dokumentację kagglehub](https://github.com/Kaggle/kagglehub).

---

## Kolejność uruchamiania komórek

Uruchom notatnik **po kolei od góry**, bo Część B korzysta z danych
przygotowanych wcześniej:

1. Sekcja 4 — importy i ustawienia.
2. Sekcja 5 — wczytanie i przygotowanie `df_sample` (wspólne).
3. Sekcja 6, Część A (6.1–6.3) — **już gotowe** (możesz uruchomić, by zobaczyć wyniki).
4. Sekcja 6, Część B (6.4–6.5) — **tu wpisujesz swój kod** (Sentiment Analysis).


