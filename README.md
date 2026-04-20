# emi-streamlit-app
# EMI Predict AI

A machine learning project that predicts **loan EMI eligibility** and the **maximum monthly EMI** an applicant can afford, based on their financial and demographic profile. The trained models are served through an interactive **Streamlit web app**.

---

## Problem Statement

Banks and NBFCs need a fast, data-driven way to assess whether a loan applicant is eligible for an EMI-based product, and if so, how much they can comfortably repay each month. This project solves two problems at once:

1. **Classification** — predict whether an applicant is `Eligible`, `High_Risk`, or `Not_Eligible`.
2. **Regression** — predict the applicant's maximum affordable monthly EMI (in rupees).

---

## Dataset

- **Size:** 400,000 rows × 27 columns (sampled down to 50,000 for training).
- **Features:** age, gender, marital status, education, monthly salary, employment type, years of employment, company type, house type, monthly rent, family size, dependents, school/college fees, travel and grocery expenses, existing loans, current EMI, credit score, bank balance, emergency fund, requested loan amount/tenure, and EMI scenario (E-commerce, Education, Home Appliances, Personal Loan, Vehicle).
- **Targets:**
  - `emi_eligibility` (classification)
  - `max_monthly_emi` (regression)
- **Target distribution:** Not_Eligible (62%), High_Risk (26%), Eligible (12%).

---

## Project Workflow

1. **Data loading & inspection** — shape, dtypes, null check, duplicate check.
2. **Exploratory Data Analysis (EDA)** — distributions of target, scenarios, education, employment, credit score, salary.
3. **Data cleaning** — drop nulls/duplicates, sample 50,000 rows for faster experimentation.
4. **Feature engineering** — added 7 derived features:
   - `total_expenses`
   - `disposable_income`
   - `debt_to_income`
   - `expense_to_income`
   - `savings_ratio`
   - `requested_emi_estimate`
   - `requested_emi_to_income`
5. **Encoding & scaling** — one-hot encoding for 7 categorical columns (`drop_first=True`), `StandardScaler` on 25 numeric columns.
6. **Train / test split** — 80 / 20 stratified split on the classification target.
7. **Classification models** — Logistic Regression, Random Forest, XGBoost.
8. **Regression models** — Linear Regression, Random Forest Regressor, XGBoost Regressor.
9. **MLflow tracking** — logged params, metrics, and models for all 6 runs under the `emi_prediction` experiment.
10. **Model deployment** — Streamlit app (`app.py`) that loads the trained artifacts and predicts for a new applicant in real time.

---

## Results

### Classification (target: `emi_eligibility`)

| Model                  | Accuracy | F1 (macro) |
|------------------------|----------|------------|
| Logistic Regression    | 0.88     | 0.84       |
| Random Forest          | 1.00     | 0.99       |
| **XGBoost (best)**     | **1.00** | **1.00**   |

### Regression (target: `max_monthly_emi`)

| Model                    | RMSE     | R²       |
|--------------------------|----------|----------|
| Linear Regression        | 3303.10  | 0.9245   |
| **Random Forest (best)** | **2.62** | **1.00** |
| XGBoost Regressor        | 113.84   | 0.9999   |

The best-performing models (XGBoost for classification and XGBoost Regressor for regression) are used by the Streamlit app.

---

## Tech Stack

- **Language:** Python 3.11+
- **Data & ML:** pandas, numpy, scikit-learn, xgboost
- **Experiment tracking:** MLflow
- **Deployment:** Streamlit
- **Serialization:** joblib
- **IDE:** VS Code (Jupyter notebook + .py scripts)

---

## Project Structure

```
EMI PREDICT AI/
├── EMI_PREDICT_AI.ipynb       # full training notebook
├── app.py                     # Streamlit web app
├── save_artifacts.py          # helper to pickle models after training
├── requirements.txt           # dependencies
├── EMI_dataset.csv            # raw dataset
├── xgb_classifier.pkl         # trained classifier
├── xgb_regressor.pkl          # trained regressor
├── scaler.pkl                 # fitted StandardScaler
├── label_encoder.pkl          # fitted LabelEncoder
├── feature_columns.pkl        # feature column order
└── README.md
```

---

## How to Run

### 1. Clone the repo and install dependencies

```bash
git clone <your-repo-url>
cd "EMI PREDICT AI"
pip install -r requirements.txt
```

### 2. Train the models (first-time only)

Open `EMI_PREDICT_AI.ipynb` in VS Code / Jupyter and run all cells. The final save cell produces the 5 `.pkl` files needed by the Streamlit app.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, fill in the applicant form in the sidebar, and click **Predict**.

### 4. (Optional) View MLflow experiments

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) to browse all six model runs with their params, metrics, and artifacts.

---

## App Preview

The Streamlit app has:
- A sidebar form with applicant details (personal, employment, housing, financial, loan request).
- A **Predict** button that runs both models.
- Output panel showing:
  - Eligibility class (color-coded: green / yellow / red).
  - Predicted maximum monthly EMI (₹).
  - Class-probability bar chart.
  - Expandable view of the preprocessed feature vector sent to the model.

---

## Key Learnings

- Strong class imbalance in `emi_eligibility` (Not_Eligible dominates) — tree-based models handled this gracefully; logistic regression lagged behind.
- The engineered ratio features (`debt_to_income`, `expense_to_income`, `requested_emi_to_income`) were the most important drivers of both predictions.
- Feature column order matters when deploying — saving the exact column list (`feature_columns.pkl`) after `pd.get_dummies` is essential to keep Streamlit predictions aligned with training.
- MLflow made it very easy to compare 6 different runs side-by-side.

---

## Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `Optuna`.
- SHAP-based explainability panel inside the app.
- Batch prediction mode (upload a CSV of applicants).
- Dockerize the app for easy cloud deployment (AWS / Azure / Streamlit Cloud).
- Add authentication for multi-user access.

---

## Author

**Megha**
Email: meghikannan@gmail.com

---

## License

This project is for educational and portfolio use.
