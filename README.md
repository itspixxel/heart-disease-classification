# Heart Disease Classification Project
> **An end-to-end Machine Learning pipeline achieving 90.16% accuracy in predicting cardiac risk.**

## Project Overview
This project leverages clinical parameters—such as age, cholesterol levels, and chest pain types—to build a predictive model for heart disease. The goal was to develop a "Proof-of-Concept" diagnostic tool that balances raw accuracy with clinical safety (minimizing missed diagnoses).

## Performance Summary
* **Champion Model:** Logistic Regression (Optimized)
* **Test Accuracy:** 90.16%
* **Cross-Validated Accuracy:** 85.46%
* **Clinical Safety (Recall):** 90.6% (Optimized via threshold adjustment)

## Tech Stack
* **Language:** Python 3.10+
* **Data Science:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-Learn`, `XGBoost`, `CatBoost`
* **Model Serialization:** `Joblib`

## Key Methodologies

### 1. Exploratory Data Analysis (EDA)
Investigated 14 clinical attributes to identify key drivers of heart disease. Significant correlations were found between heart disease and:
* **Chest Pain Type (`cp`)**: Atypical symptoms often correlated higher with disease presence.
* **Max Heart Rate (`thalach`)**: A reliable physiological stress indicator.
* **Vessel Count (`ca`)**: Number of major vessels colored by fluoroscopy.

### 2. Feature Engineering & Preprocessing
* **One-Hot Encoding:** Categorical features (like `cp` and `thal`) were encoded to remove ordinal bias, providing a significant performance boost over baseline models.
* **Hyperparameter Tuning:** Utilized `GridSearchCV` to optimize the regularization strength (`C`) of the Logistic Regression model, ensuring better generalization.

### 3. Medical Risk Optimization
In a clinical setting, **Recall** is more critical than Accuracy. I implemented a custom decision threshold of **0.45** (down from the default 0.50), which successfully reduced False Negatives, ensuring more high-risk patients are correctly identified.

## Repository Structure
```bash
├── data/
│   └── heart-disease.csv          # Cleveland UCI Dataset
├── models/
│   └── heart_disease_model.pkl    # Serialized Champion Model
│   └── model_columns.pkl          # Feature column order for inference
├── notebooks/
│   └── heart-disease-classification.ipynb # Full analysis and code
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies

```

## How to Run

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/heart-disease-classification.git](https://github.com/your-username/heart-disease-classification.git)

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the analysis:**
Open the Jupyter Notebook in the `notebooks/` directory to view the full pipeline and experimentation.

## Future Work

* **Interaction Features:** Developing features that combine age and heart rate metrics to capture non-linear relationships.
* **Explainability:** Integrating SHAP or LIME to provide interpretability for clinical decision support.
* **Deployment:** Wrapping the model in a FastAPI or Streamlit application for real-time predictions.