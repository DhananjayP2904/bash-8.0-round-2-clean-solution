# bash-8.0-round-2-clean-solution

# Epstein Legal Documents – Importance Score Prediction

This repository contains a clean and reproducible solution for the
BASH 8.0 Round 2 competition.

## Approach Overview

The solution treats the problem as a supervised regression task.
The model predicts an "Importance Score" for each document based on
textual and structural information provided in the dataset.

### Key Design Principles
- No hard-coded keywords or domain-specific rules
- All features are learned automatically from data
- Same processing pipeline for train and test
- Cross-validation used to avoid overfitting

## Features Used

### Textual Features
- Headline
- Key Insights
- Reasoning

All text fields are combined and processed using:
- TF-IDF (unigrams + bigrams)
- Truncated SVD for semantic compression

### Structured Features
- Count of lead types
- Count of agencies
- Count of power mentions
- Count of tags

### Optional Semantic Embeddings
- Sentence embeddings generated using a pretrained SentenceTransformer
- Dimensionality reduced with PCA

## Models
- LightGBM Regressor
- XGBoost Regressor
- Ridge regression used as a stacking meta-learner

## Evaluation
- 5-fold cross-validation
- RMSE used as the evaluation metric

## Data
The dataset is provided by the competition and is not included in this repository.
To run the code, place `train.csv` and `test.csv` inside a `data/` directory.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
