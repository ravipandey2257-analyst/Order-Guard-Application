# Order Guard — Streamlit + AdaBoost

A Streamlit application for predicting Amazon order cancellation risk using AdaBoost.

## Run locally

1. Create/activate a Python virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Upload `Amazon Sale Report.csv` in the sidebar.

## Expected training columns

- `Status` (the app creates `Cancelled = 1` when Status is `Cancelled`, otherwise 0), or an existing binary `Cancelled` column
- `Fulfilment`
- `Sales Channel`
- `ship-service-level`
- `Category`
- `Size`
- `ship-state`
- `B2B`
- `Amount`
- `Date` (used to derive month and day of week if those fields are not already present)

## Model

The app uses `AdaBoostClassifier(n_estimators=100, random_state=42)`, matching the main AdaBoost configuration in the supplied notebook. It uses a scikit-learn preprocessing pipeline with imputation and one-hot encoding, plus balanced sample weights during training. The model is trained when the dataset is uploaded; no prebuilt pickle is required.

## Important

Predictions are estimates, not guarantees. Validate performance on representative, current data before using the model operationally. The selected probability threshold changes the high-risk flag and predicted class.
