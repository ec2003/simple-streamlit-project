'''
STREAMLIT DEPLOYMENT URL 
https://ec2003-heart-disease-detection.streamlit.app/
'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient clinical data below to predict heart disease probability across multiple ML models.")

# ---------------------------------------------------------------------------
# Real-world ranges for each feature (for UI display)
# ---------------------------------------------------------------------------
REAL_RANGES = {
    'age':       (29, 77),
    'trestbps':  (94, 200),
    'chol':      (126, 564),
    'thalach':   (71, 202),
    'oldpeak':   (0.0, 6.2),
}

# Categorical feature mappings: user-facing label (int) -> scaled value
CAT_MAP = {
    'cp':      {0: 0.0, 1: 1/3, 2: 2/3, 3: 1.0},
    'restecg': {0: 0.0, 1: 0.5, 2: 1.0},
    'slope':   {0: 0.0, 1: 0.5, 2: 1.0},
    'ca':      {0: 0.0, 1: 1/3, 2: 2/3, 3: 1.0},
    'thal':    {0: 0.0, 1: 0.75, 2: 1.0},
}

# Binary features that pass through unchanged
BINARY_FEATURES = ['sex', 'fbs', 'exang']

# ---------------------------------------------------------------------------
# Load data & models
# ---------------------------------------------------------------------------
@st.cache_data
def load_training_data():
    return pd.read_csv("dataset/raw_train.csv")

@st.cache_resource
def load_models():
    model_dir = "ml_models"
    models = {}
    model_files = {
        "Decision Tree":        "decision_tree.pkl",
        "AdaBoost":             "adaboost.pkl",
        "Random Forest":        "random_forest.pkl",
        "Gradient Boosting":    "gradient_boosting.pkl",
        "XGBoost":              "xgboost.pkl",
        "k-NN":                 "k-nn.pkl",
        "Naïve Bayes":          "naïve_bayes.pkl",
        "Ensemble (Soft Voting)": "ensemble_soft_voting.pkl",
    }
    for display_name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[display_name] = joblib.load(path)
    return models

@st.cache_data
def load_performance():
    path = "ml_models/model_performance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

train_df = load_training_data()
models = load_models()
perf_df = load_performance()
feature_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Scaled min/max computed from the pre-scaled training data
SCALED_RANGES = {}
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    SCALED_RANGES[col] = (float(train_df[col].min()), float(train_df[col].max()))

# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------
def real_to_scaled(real_val, col):
    """Convert a real-world value to the scaled value the models expect."""
    if col in BINARY_FEATURES:
        return float(real_val)
    if col in CAT_MAP:
        return CAT_MAP[col][int(real_val)]
    # Continuous feature – linear min-max rescale
    r_min, r_max = REAL_RANGES[col]
    s_min, s_max = SCALED_RANGES[col]
    if r_max == r_min:
        return s_min
    return s_min + (real_val - r_min) / (r_max - r_min) * (s_max - s_min)

def scaled_to_real(scaled_val, col):
    """Reverse mapping (for displaying example patients)."""
    if col in BINARY_FEATURES:
        return int(round(scaled_val))
    if col in CAT_MAP:
        # Find closest integer key
        inv = {round(v, 4): k for k, v in CAT_MAP[col].items()}
        return inv.get(round(scaled_val, 4), 0)
    r_min, r_max = REAL_RANGES[col]
    s_min, s_max = SCALED_RANGES[col]
    if s_max == s_min:
        return r_min
    return round(r_min + (scaled_val - s_min) / (s_max - s_min) * (r_max - r_min), 2)

# ---------------------------------------------------------------------------
# Build example patients (store both real and scaled)
# ---------------------------------------------------------------------------
def build_example(scaled_row):
    return {col: scaled_to_real(scaled_row[col], col) for col in feature_cols}

example_no_hd_row = train_df[train_df['target'] == 0].iloc[0]
example_hd_row   = train_df[train_df['target'] == 1].iloc[0]

examples = {
    "Custom": None,
    "Example 1 (No Heart Disease)": build_example(example_no_hd_row),
    "Example 2 (Heart Disease)":    build_example(example_hd_row),
}

# ---------------------------------------------------------------------------
# Sidebar input panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Feature Input")

    selected_example = st.selectbox("Select Example Patient", list(examples.keys()))

    if selected_example == "Custom":
        default_real = {col: 0 for col in feature_cols}
        # Sensible neutral defaults for continuous
        default_real['age']      = 50
        default_real['trestbps'] = 130
        default_real['chol']     = 250
        default_real['thalach']  = 140
        default_real['oldpeak']  = 1.0
    else:
        default_real = examples[selected_example]

    st.markdown("---")
    st.subheader("Demographics")

    age = st.slider("Age", 29, 77, int(default_real['age']))
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                   index=int(default_real['sex']))

    st.subheader("Vital Signs & Lab Results")

    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, int(default_real['trestbps']))
        chol     = st.slider("Serum Cholesterol (mg/dl)", 126, 564, int(default_real['chol']))
    with col2:
        thalach  = st.slider("Max Heart Rate Achieved", 71, 202, int(default_real['thalach']))
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, float(default_real['oldpeak']), step=0.1)

    st.subheader("Test Results")

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x],
        index=int(default_real['cp'])
    )

    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                   format_func=lambda x: "No" if x == 0 else "Yes",
                   index=int(default_real['fbs']))

    restecg = st.selectbox(
        "Resting ECG Results",
        [0, 1, 2],
        format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
        index=int(default_real['restecg'])
    )

    exang = st.radio("Exercise Induced Angina", [0, 1],
                     format_func=lambda x: "No" if x == 0 else "Yes",
                     index=int(default_real['exang']))

    slope = st.selectbox(
        "ST Segment Slope",
        [0, 1, 2],
        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
        index=int(default_real['slope'])
    )

    ca = st.selectbox(
        "Number of Major Vessels (0-3)",
        [0, 1, 2, 3],
        index=int(default_real['ca'])
    )

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2],
        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x],
        index=int(default_real['thal'])
    )

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict", type="primary")

# ---------------------------------------------------------------------------
# Main area – prediction results
# ---------------------------------------------------------------------------
if predict_clicked and models:
    # Collect real-world values from widgets
    real_values = {
        'age': age, 'trestbps': trestbps, 'chol': chol,
        'thalach': thalach, 'oldpeak': oldpeak, 'sex': sex,
        'cp': cp, 'fbs': fbs, 'restecg': restecg,
        'exang': exang, 'slope': slope, 'ca': ca, 'thal': thal,
    }

    # Scale to model inputs
    scaled_values = {col: real_to_scaled(real_values[col], col) for col in feature_cols}
    input_df = pd.DataFrame([scaled_values])

    # Get predictions from all models
    predictions = []
    for model_name, model in models.items():
        pred_class = model.predict(input_df)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            prob_no_hd, prob_hd = proba[0], proba[1]
        else:
            prob_no_hd = 1.0 - pred_class
            prob_hd    = pred_class

        predictions.append({
            'Model': model_name,
            'Prediction': "No Heart Disease" if pred_class == 0 else "Heart Disease",
            'No HD Prob': prob_no_hd,
            'HD Prob': prob_hd,
        })

    pred_df = pd.DataFrame(predictions)

    # ----- Bar chart -----
    st.subheader("📊 Model Predictions Overview")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(pred_df))
    bar_h = 0.35

    ax.barh(y_pos - bar_h/2, pred_df['No HD Prob'], bar_h,
            label='No Heart Disease', color='#2ecc71', alpha=0.85)
    ax.barh(y_pos + bar_h/2, pred_df['HD Prob'], bar_h,
            label='Heart Disease', color='#e74c3c', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pred_df['Model'], fontsize=10)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Model Predictions Overview', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    for i, (_, row) in enumerate(pred_df.iterrows()):
        if row['HD Prob'] > row['No HD Prob']:
            ax.text(row['HD Prob'] + 0.02, i + bar_h/2, 'Disease',
                    va='center', fontsize=8, color='#e74c3c', fontweight='bold')
        else:
            ax.text(row['No HD Prob'] + 0.02, i - bar_h/2, 'No Disease',
                    va='center', fontsize=8, color='#2ecc71', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # ----- Detail table -----
    st.subheader("📋 Detailed Predictions")
    display_df = pred_df.copy()
    display_df['No HD Prob'] = display_df['No HD Prob'].apply(lambda x: f"{x:.1%}")
    display_df['HD Prob']    = display_df['HD Prob'].apply(lambda x: f"{x:.1%}")
    display_df.columns = ['Model', 'Prediction', 'No HD Probability', 'HD Probability']

    def color_pred(val):
        return 'color: #ffe0e0' if val == "Heart Disease" else 'color: #e0ffe0'

    styled_df = display_df.style.map(color_pred, subset=['Prediction'])
    st.dataframe(styled_df, width='stretch', hide_index=True)

else:
    st.info("👈 Enter patient data in the sidebar and click **Predict** to see results.")

    if perf_df is not None:
        st.subheader("📈 Model Performance on Validation Set")
        st.dataframe(perf_df, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Models Used")
    st.markdown("""
    - **Decision Tree Classifier**
    - **AdaBoost Classifier**
    - **Random Forest Classifier**
    - **Gradient Boosting Classifier**
    - **XGBoost Classifier**
    - **k-Nearest Neighbors (k-NN)**
    - **Naïve Bayes (GaussianNB)**
    - **Ensemble (Soft Voting)** — aggregates all 7 models
    """)