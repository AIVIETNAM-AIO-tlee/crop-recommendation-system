# Crop Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

## Topic

Crop Recommendation System using distance-based retrieval and K-Means++ accelerated KNN.

## Brief Topic Description

This project recommends the most suitable crop from soil and weather measurements.
It compares a full-search KNN baseline with a K-Means++ indexed KNN approach to balance recommendation quality and inference speed.
The system is designed for Top-K crop recommendation rather than single-class classification.

## Input

The model uses 7 numerical features:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

## Output

The system returns a ranked Top-K list of crop labels, for example:

- rice
- maize
- cotton
- jute
- coffee

It also stores evaluation outputs such as MAP@K, NDCG@K, HitRate@K, latency, and candidate reduction.

## Team

| Name                | Role                                |
| ------------------- | ----------------------------------- |
| Thanh Le Quang      | AI Engineer (pipeline), Team Leader |
| Vo Ngoc Gia Bao     | QA/Reviewer                         |
| Van Vi Nguyen       | AI Engineer (model)                 |
| Thanh Lam Le        | Tech Leader                         |
| Pham Minh Dang Tran | AI Engineer (data)                  |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow (branching, pull requests, coding guidelines, local setup). Summary of what each member contributed:

- **Thanh Le Quang** (AI Engineer – pipeline, Team Leader): project kickoff and team alignment, Jira/Kanban setup, repository folder structure and pipeline scaffolding, evaluation metrics module, Streamlit demo app, demo video, final review and retro.
- **Thanh Lam Le** (Tech Leader): GitHub repository and README setup, K-Means++ accelerated KNN (KNN+) search implementation, presentation slides and script.
- **Pham Minh Dang Tran** (AI Engineer – data): data collection and sourcing, exploratory data analysis, preprocessing (missing values, scaling), PCA and train/val/test splitting.
- **Van Vi Nguyen** (AI Engineer – model): KNN baseline implementation and distance metrics.
- **Vo Ngoc Gia Bao** (QA/Reviewer): validation and review at every stage of the project — folder structure and data sourcing, data pipeline (EDA, preprocessing, PCA), model and evaluation metrics, and the final end-to-end QA report.

## Dataset

The project uses the crop recommendation dataset stored in `data/raw/Crop_recommendation.csv`.

- Source format: tabular CSV
- Target column: `label`
- Feature columns: the 7 soil and weather variables listed above
- Split files: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

## Repository Structure

```text
crop-recommendation-system/
├── app/                         # Streamlit application and pages
├── data/
│   ├── raw/                     # Original CSV dataset
│   └── processed/               # Train, validation, and test CSV files
├── docs/                        # Course documents, Jira exports, and meeting notes
├── models/                      # Saved imputer, scaler, and trained recommenders
├── notebooks/
│   └── main/                    # Main experiment notebook
├── reports/
│   ├── EXPERIMENT_UPDATE.md     # Methodology and findings
│   └── tables/                  # Validation, test, CI, latency, and label tables
├── src/
│   ├── config.py                # Shared experiment configuration
│   ├── main.py                  # End-to-end experiment entry point
│   ├── data/                    # Loading, preprocessing, and splitting
│   ├── evaluation/              # Metrics, bootstrap, latency, tuning, ground truth
│   └── models/                  # Exact KNN and K-Means++-indexed KNN
├── tests/                       # Unit and integration tests
├── requirements.txt             # Pinned dependencies
├── pytest.ini
├── CONTRIBUTING.md
└── README.md
```

## How to Run

Train and save the final artifacts:

```bash
cd src
python -m pip install -r requirements.txt
python main.py
```

Launch the Streamlit app:

```bash
streamlit run app/app.py
```

## Notes

- The final artifacts are saved in `models/` after training.
- The evaluation reports are saved in `reports/tables/`.
- The Streamlit app reads the saved artifacts and report files directly.

## Reproducibility Note

The final evaluation workflow includes component-level latency profiling with a configurable p95 threshold, 95% bootstrap confidence intervals for MAP@5 and NDCG@5, paired bootstrap comparison of Exact KNN and KNN + K-Means++, weak 2--3 crop ground truth, unit and integration tests, and pinned dependency versions.

See [reports/EXPERIMENT_UPDATE.md](reports/EXPERIMENT_UPDATE.md) for the methodology and results.
