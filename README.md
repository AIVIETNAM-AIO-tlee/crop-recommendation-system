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

## Dataset

The project uses the crop recommendation dataset stored in `data/raw/Crop_recommendation.csv`.

- Source format: tabular CSV
- Target column: `label`
- Feature columns: the 7 soil and weather variables listed above
- Split files: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

## Folder Structure

```text
crop-recommendation-system/
├── data/
│   ├── raw/
│   │   └── Crop_recommendation.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── README.md
├── models/
│   ├── final_imputer.pkl
│   ├── final_scaler.pkl
│   ├── baseline_best_model.pkl
│   ├── hybrid_best_model.pkl
│   └── artifact_metadata.json
├── notebooks/
│   ├── main/
│   │   └── Module2_Crop_recommendation_system.ipynb
│   ├── notebook_1/
│   │   └── Dang_Tran_Crop_Recommendation_Final.ipynb
│   ├── notebook_2/
│   │   └── Distance_metrics_and_similarity_design_for_recommendation_systems.ipynb
│   └── notebook_3/
│       └── Module2_K_Means__KNN.ipynb
├── src/
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── split.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── tuning.py
│   └── models/
│       ├── base_knn.py
│       └── kmeanSearch_knn.py
├── app/
│   ├── app.py
│   └── pages/
│       ├── 1_Recommendation.py
│       ├── 2_Model_Comparison.py
│       └── 3_Dataset.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── report.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

## Notebook Organization

The `notebooks/main/` folder contains the single main notebook used as the central workflow for the project: `Module2_Crop_recommendation_system.ipynb`.

The other notebook folders are separate working notebooks prepared by the three team members:

- `notebooks/notebook_1/` 
- `notebooks/notebook_2/` 
- `notebooks/notebook_3/`

## How to Run

Train and save the final artifacts:

```bash
cd src
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
