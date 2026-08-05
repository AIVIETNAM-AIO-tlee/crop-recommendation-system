# Contributing to Crop Recommendation System

Thanks for your interest in this project. This document describes how the team works, how the codebase is organized, and how to set up, run, and contribute changes.

## Team and Roles

| Name | Role | Jira focus area |
| --- | --- | --- |
| Thanh Le Quang | AI Engineer (pipeline), Team Leader | Project setup, pipeline/folder structure, demo app, team coordination |
| Thanh Lam Le | Tech Leader | GitHub repo setup, KNN + K-Means++ search, presentation |
| Pham Minh Dang Tran | AI Engineer (data) | Data collection, EDA, preprocessing, PCA and data splitting |
| Van Vi Nguyen | AI Engineer (model) | KNN baseline, distance metrics, evaluation metrics |
| Vo Ngoc Gia Bao | QA / Reviewer | Validating each stage (folder structure, data pipeline, model/metrics, final end-to-end QA) |

Work is planned and tracked on the team's Jira board (project **ACM2 — AIO-CONQUER MODULE 2**). The board is organized into four epics, each owned by one engineer and independently validated by QA:

1. **[EPIC 1] Project Init & Team Alignment** — kickoff, Kanban/Jira setup, GitHub + README, folder structure, initial data sourcing.
2. **[EPIC 2] Data Pipeline & Preprocessing** — EDA, missing-value handling, scaling, PCA, train/val/test split.
3. **[EPIC 3] Build Model, Distance & Evaluation Metrics** — KNN baseline, K-Means++-accelerated KNN, distance metrics, ranking evaluation metrics (Precision@K, MAP@K, NDCG@K, etc.).
4. **[EPIC 4] Demo & Final Delivery** — Streamlit demo app, demo video, presentation slides, final QA report, retro.

Every epic includes a `[QA]` task where Vo Ngoc Gia Bao validates the work before it is considered done — please keep this pattern for any new epic or major feature: implementation task(s) + a QA/validation task.

## Getting Started

```bash
git clone https://github.com/AIVIETNAM-AIO-tlee/crop-recommendation-system.git
cd crop-recommendation-system
python -m pip install -r requirements.txt
```

Run the training/evaluation pipeline:

```bash
PYTHONPATH=src python src/main.py
```

Run the test suite (required before opening a PR):

```bash
pytest
```

Launch the Streamlit demo:

```bash
streamlit run app/app.py
```

Generated evidence (latency profiling, bootstrap confidence intervals, weak multi-label ground truth, etc.) is written to `reports/tables/`; the methodology write-up lives in `reports/EXPERIMENT_UPDATE.md`.

## Branching and Pull Requests

- `main` is the protected, always-runnable branch. Do not commit directly to `main`.
- Work on a personal or feature branch (the team currently uses per-member branches such as `Dang`, `Lam`, `vi`; feature-scoped branches like `feature/<short-description>` are also welcome for larger changes).
- Keep commits scoped and use clear messages, e.g. `fix: update path in notebook_2`, `feat: add display best model for validation in 1_Recommendation`.
- Open a **Compare & pull request** into `main` once your branch is ready. Reference the related Jira issue (e.g. `ACM2-16`) in the PR description.
- Make sure `pytest` passes locally before requesting review.
- At least one other contributor (ideally QA) should review and approve before merging.

## Coding Guidelines

- Python 3.10+, keep dependencies pinned in `requirements.txt` if you add a new package.
- New evaluation logic (metrics, bootstrap analysis, latency profiling) should be reproducible from `src/main.py` / `src/evaluation/` and covered by a corresponding test in `tests/`.
- Avoid hardcoding local paths (e.g. personal Google Drive paths); use the existing `src/config.py` / relative-path conventions.
- If your change affects reported numbers (latency, MAP@K, NDCG@K, bootstrap CIs), regenerate the tables in `reports/tables/` and note it in `reports/EXPERIMENT_UPDATE.md`.

## Reporting Issues / Proposing Work

- Use the Jira board (project ACM2) to file new tasks or bugs, and link the corresponding GitHub PR back to the Jira issue.
- For quick discussion or small fixes without a tracked Jira item, open a GitHub Issue describing the problem and, if known, the affected file(s).

## Questions

For questions about the pipeline/architecture, reach out to Thanh Le Quang or Thanh Lam Le; for data/preprocessing, Pham Minh Dang Tran; for model/evaluation, Van Vi Nguyen; for QA/validation process, Vo Ngoc Gia Bao.