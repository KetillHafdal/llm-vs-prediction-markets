# LLMs vs Prediction Markets

This repository contains the data collection, processing, and analysis code used in the Master's thesis:

> *To what extent are large language models able to replicate or complement human forecasting in decentralised prediction markets?*

The thesis evaluates the forecasting performance of large language models relative to market-implied probabilities from Polymarket.

## Repository structure

- `src/collection/` – scripts for collecting Polymarket data and generating LLM forecasts  
- `src/processing/` – data cleaning and aggregation scripts  
- `analysis/` – RMarkdown files used for statistical analysis  
- `data/processed/` – processed datasets used in the analysis  
- `docs/` – replication and documentation files  

## Data availability

Due to third-party API restrictions (e.g. AskNews), raw retrieved text data are not included in this repository.  
Scripts are provided to re-run data collection given appropriate API access.

## Reproducibility

Instructions for reproducing the analysis are provided in `docs/replication.md`.
