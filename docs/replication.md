# Replication instructions

This document describes how to reproduce the data collection and analysis pipeline used in the thesis.

The repository is intended as a replication package rather than a fully automated application. Several steps must be executed repeatedly over time.

---

## 1. Requirements

### Software
- Python (for data collection and forecasting)
- R (for statistical analysis)

### API access
Replication requires access to the following third-party APIs:
- OpenRouter (LLM inference)
- AskNews (news retrieval)

API credentials are not included in this repository.

---

## 2. Environment setup

Create a local `.env` file in the repository root and populate it using `.env.example` as a template:

```
OPENROUTER_API_KEY=your_key_here
ASKNEWS_CLIENT_ID=your_id_here
ASKNEWS_SECRET=your_secret_here
```

The `.env` file is ignored by Git and must not be committed.

---

## 3. Data collection pipeline

### 3.1 Market universe definition (run once)

To define the universe of Polymarket markets analysed in the thesis, run:

```
python src/collection/fetch_polymarket_initial_markets.py
```

This script retrieves the set of markets that form the basis of the empirical analysis.

---

### 3.2 Daily data collection (run daily)

The following scripts are intended to be executed once per day over the data collection period.

#### Polymarket daily snapshot

```
python src/collection/fetch_polymarket_daily_snapshot.py
```

This script collects a cross-sectional snapshot of market prices for all active markets on the given day.

#### LLM forecasts

```
python src/collection/run_llm_forecasts.py
```

This script retrieves external information via the AskNews API and generates probabilistic forecasts using large language models accessed through OpenRouter.

Both scripts must be run on the same day to ensure temporal alignment between market prices and LLM forecasts.

---

### 3.3 Market resolutions (run after resolution)

After markets have resolved, realised outcomes can be retrieved using:

```
python src/collection/fetch_polymarket_resolutions.py
```

These outcomes serve as ground truth for forecast evaluation.

---

## 4. Analysis

Statistical analysis and result generation are performed in R using:

```
analysis/main_analysis.Rmd
```

This file loads the processed datasets, computes forecast accuracy metrics (e.g. Brier scores), performs statistical tests, and produces the figures and tables reported in the thesis.

Rendered outputs are not committed by default.

---

## 5. Data availability and limitations

- Raw retrieved news article text from AskNews is not included due to third-party API restrictions.
- LLM outputs are stored only in derived form (forecast probabilities).
- Reproducibility depends on API access, execution timing, and market availability.

As a result, exact numerical replication may not be possible without matching data collection conditions.
