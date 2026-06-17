# Heart Disease Prediction App

A Streamlit web application that predicts heart disease probability using multiple machine learning models. The app takes patient clinical data as input and visualizes predictions from 8 different ML models in an interactive dashboard.

## Features

- **Real-world patient inputs** — Age, blood pressure, cholesterol, chest pain type, and other clinical metrics with human-readable values
- **8 ML models** — Decision Tree, AdaBoost, Random Forest, Gradient Boosting, XGBoost, k-Nearest Neighbors, Naïve Bayes, and Ensemble Soft Voting
- **Interactive visualization** — Horizontal bar chart showing probability scores with green (normal) / red (heart disease) color coding
- **Model performance table** — Validation accuracy displayed for all models
- **Example patients** — Quick-load preset cases for demonstration

## Dataset

The models are trained on the **UCI Heart Disease dataset**, pre-split into:

| Split       | Samples |
|-------------|---------|
| Training    | 242     |
| Validation  | 30      |
| Test        | 31      |

The data is located in the `dataset/` folder as CSV files with pre-scaled feature values.

## Installation

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/ec2003/simple-streamlit-project.git
cd simple-streamlit-project

# Install dependencies
uv sync
```

## Usage

### Train models (optional)

If you want to retrain the models from scratch:

```bash
# Open and run the Jupyter notebook
uv run jupyter notebook notebooks/train_models.ipynb
```

### Run the app

```bash
uv run streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## Models

| Model                    | Validation Accuracy |
|--------------------------|---------------------|
| Decision Tree            | ~76.7%              |
| AdaBoost                 | ~90.0%              |
| Random Forest            | ~93.3%              |
| Gradient Boosting        | ~90.0%              |
| XGBoost                  | ~93.3%              |
| k-Nearest Neighbors (k-NN)| ~93.3%              |
| Naïve Bayes (GaussianNB) | ~90.0%              |
| Ensemble (Soft Voting)   | ~93.3%              |

Trained models are saved as `.pkl` files in the `ml_models/` directory.

## Project Structure

```
├── app.py                     # Streamlit application
├── pyproject.toml             # Project dependencies
├── main.py                    # Entry point (legacy)
├── README.md                  # This file
├── dataset/
│   ├── raw_train.csv          # Training data
│   ├── raw_val.csv            # Validation data
│   └── raw_test.csv           # Test data
├── ml_models/
│   ├── decision_tree.pkl
│   ├── adaboost.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── xgboost.pkl
│   ├── k-nn.pkl
│   ├── naïve_bayes.pkl
│   ├── ensemble_soft_voting.pkl
│   └── model_performance.csv
└── notebooks/
    └── train_models.ipynb     # Model training notebook
```

## Deployment

The app is also deployed on Streamlit Community Cloud:

[https://ec2003-heart-disease-detection.streamlit.app/](https://ec2003-heart-disease-detection.streamlit.app/)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.