import io
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight

st.set_page_config(
    page_title="Order Guard | Cancellation Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px;
       background: linear-gradient(115deg,#14213d,#263f68); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2.1rem;}
.hero p {color:#dce6f5; margin:.45rem 0 0 0;}
[data-testid="stMetric"] {background: rgba(120,140,170,.08); padding: 14px; border-radius: 12px;}
.risk-high {padding:1rem; border-radius:12px; background:#3b171b; border:1px solid #a83e49;}
.risk-low {padding:1rem; border-radius:12px; background:#143426; border:1px solid #2c8b61;}
.small-note {color: #8b95a5; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📦 Order Guard</h1>
  <p>Amazon order cancellation prediction powered by AdaBoost</p>
</div>
""", unsafe_allow_html=True)

FEATURES = [
    "Fulfilment", "Sales Channel", "ship-service-level", "Category",
    "Size", "ship-state", "B2B", "Amount", "month", "dayofweek"
]
CAT_COLS = [
    "Fulfilment", "Sales Channel", "ship-service-level",
    "Category", "Size", "ship-state"
]
NUM_COLS = ["B2B", "Amount", "month", "dayofweek"]
TARGET = "Cancelled"

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

def prepare_dataset(raw):
    df = raw.copy()
    # Accept either the notebook's original Status target or an already-created target.
    if TARGET not in df.columns:
        if "Status" not in df.columns:
            raise ValueError("The training CSV must contain a 'Status' column (or a 'Cancelled' column).")
        df[TARGET] = (df["Status"].astype(str).str.strip().str.lower() == "cancelled").astype(int)
    else:
        if df[TARGET].dtype == object:
            mapped = df[TARGET].astype(str).str.strip().str.lower().map({
                "1": 1, "true": 1, "yes": 1, "cancelled": 1,
                "0": 0, "false": 0, "no": 0, "not cancelled": 0
            })
            df[TARGET] = mapped
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Derive the calendar fields used in the source notebook.
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], format="%m-%d-%y", errors="coerce")
        # Also handle common alternative date formats.
        missing = dates.isna()
        if missing.any():
            dates.loc[missing] = pd.to_datetime(df.loc[missing, "Date"], errors="coerce")
        if "month" not in df.columns:
            df["month"] = dates.dt.month
        if "dayofweek" not in df.columns:
            df["dayofweek"] = dates.dt.dayofweek

    required = FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    df = df[required].copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("Unknown").astype(str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["B2B"] = df["B2B"].fillna(0)
    df[TARGET] = df[TARGET].fillna(0).astype(int)
    df = df[df[TARGET].isin([0, 1])]
    if df[TARGET].nunique() != 2:
        raise ValueError("The training data must contain both cancelled and non-cancelled orders.")
    return df

def build_model():
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    prep = ColumnTransformer(
        transformers=[
            ("categorical", categorical, CAT_COLS),
            ("numeric", numeric, NUM_COLS)
        ],
        remainder="drop"
    )
    # Same core AdaBoost configuration used in the notebook.
    classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
    return Pipeline([("preprocess", prep), ("adaboost", classifier)])

def normalize_prediction_input(frame):
    x = frame.copy()
    for c in CAT_COLS:
        if c not in x.columns:
            x[c] = "Unknown"
        x[c] = x[c].fillna("Unknown").astype(str)
    for c in NUM_COLS:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[FEATURES]

with st.sidebar:
    st.header("⚙️ Model setup")
    st.caption("Upload the Amazon Sale Report CSV used by the notebook.")
    training_file = st.file_uploader("Training dataset (CSV)", type=["csv"], key="train_csv")
    st.divider()
    st.markdown("**Algorithm**")
    st.info("AdaBoost Classifier\n\n100 estimators · random_state=42")
    st.markdown("**Target**")
    st.caption("Cancelled = 1 · Not cancelled = 0")

if training_file is None:
    st.info("Upload your **Amazon Sale Report.csv** in the sidebar to train the AdaBoost model and unlock predictions.")
    st.markdown("### Expected dataset columns")
    st.code(", ".join(FEATURES + ["Status"]), language="text")
    st.markdown("""
    **How to run**
    1. Install dependencies: `pip install -r requirements.txt`
    2. Start the app: `streamlit run app.py`
    3. Upload your Amazon sales CSV in the sidebar.
    """)
    st.stop()

try:
    raw_df = load_csv(training_file.getvalue())
    data = prepare_dataset(raw_df)
except Exception as exc:
    st.error(f"Could not prepare the training dataset: {exc}")
    st.stop()

if len(data) < 10:
    st.error("At least 10 valid rows are recommended for a train/test split.")
    st.stop()

X = data[FEATURES]
y = data[TARGET]
test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
threshold = st.sidebar.slider("Cancellation risk threshold", 0.10, 0.90, 0.50, 0.05)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = build_model()
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    with st.spinner("Training AdaBoost model..."):
        model.fit(X_train, y_train, adaboost__sample_weight=sample_weights)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, test_prob)
except Exception:
    auc = float("nan")

st.success(f"AdaBoost trained successfully on {len(X_train):,} orders.")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total orders", f"{len(data):,}")
m2.metric("Cancellation rate", f"{y.mean()*100:.1f}%")
m3.metric("Accuracy", f"{accuracy*100:.1f}%")
m4.metric("Cancelled recall", f"{recall*100:.1f}%")
m5.metric("F1 score", f"{f1*100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔎 Single prediction", "📁 Batch prediction", "🧪 Model evaluation"
])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Order outcome distribution")
        counts = data[TARGET].map({0: "Not cancelled", 1: "Cancelled"}).value_counts()
        st.bar_chart(counts)
    with right:
        st.subheader("Dataset preview")
        st.dataframe(data.head(12), use_container_width=True)
    st.caption("Cancellation label is derived from Status == 'Cancelled' when the CSV does not already contain a Cancelled column.")

with tab2:
    st.subheader("Predict cancellation risk for one order")
    st.caption("Choose values that describe the order. Unknown categories are safely handled by the trained encoder.")
    c1, c2, c3 = st.columns(3)
    categories = {c: sorted(data[c].dropna().astype(str).unique().tolist()) for c in CAT_COLS}
    with c1:
        fulfilment = st.selectbox("Fulfilment", categories["Fulfilment"])
        sales_channel = st.selectbox("Sales Channel", categories["Sales Channel"])
        service = st.selectbox("Shipping service level", categories["ship-service-level"])
        category = st.selectbox("Category", categories["Category"])
    with c2:
        size = st.selectbox("Size", categories["Size"])
        ship_state = st.selectbox("Shipping state", categories["ship-state"])
        b2b = st.selectbox("B2B order?", ["No", "Yes"])
        amount = st.number_input("Order amount", min_value=0.0, value=float(data["Amount"].median() if pd.notna(data["Amount"].median()) else 0), step=50.0)
    with c3:
        month = st.selectbox("Order month", list(range(1, 13)), index=3)
        dayofweek = st.selectbox("Day of week (0=Mon, 6=Sun)", list(range(7)), index=5)
    if st.button("🔮 Predict order risk", type="primary", use_container_width=True):
        one = pd.DataFrame([{
            "Fulfilment": fulfilment, "Sales Channel": sales_channel,
            "ship-service-level": service, "Category": category, "Size": size,
            "ship-state": ship_state, "B2B": 1 if b2b == "Yes" else 0,
            "Amount": amount, "month": month, "dayofweek": dayofweek
        }])
        p = float(model.predict_proba(one[FEATURES])[0, 1])
        label = "HIGHER CANCELLATION RISK" if p >= threshold else "LOWER CANCELLATION RISK"
        css = "risk-high" if p >= threshold else "risk-low"
        st.markdown(f'<div class="{css}"><h3>{label}</h3><h2>{p*100:.1f}% cancellation probability</h2><p>Threshold: {threshold*100:.0f}%</p></div>', unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0))
        st.caption("This is a model estimate, not a guarantee that an order will or will not be cancelled.")

with tab3:
    st.subheader("Score many orders at once")
    batch_file = st.file_uploader("Upload orders CSV", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            batch_raw = load_csv(batch_file.getvalue())
            batch_x = normalize_prediction_input(batch_raw)
            batch_prob = model.predict_proba(batch_x)[:, 1]
            results = batch_raw.copy()
            results["Cancellation_Probability"] = batch_prob
            results["Predicted_Cancelled"] = (batch_prob >= threshold).astype(int)
            results["Predicted_Status"] = np.where(results["Predicted_Cancelled"] == 1, "Cancelled risk", "Not cancelled risk")
            st.dataframe(results.head(100), use_container_width=True)
            st.metric("Orders flagged as higher risk", f"{int((results['Predicted_Cancelled'] == 1).sum()):,}")
            st.download_button(
                "⬇️ Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="order_guard_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")

with tab4:
    st.subheader("Held-out test set performance")
    a, b, c = st.columns(3)
    a.metric("Precision", f"{precision*100:.1f}%")
    b.metric("F1 score", f"{f1*100:.1f}%")
    c.metric("ROC AUC", "N/A" if np.isnan(auc) else f"{auc:.3f}")
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual not cancelled", "Actual cancelled"],
                              columns=["Predicted not cancelled", "Predicted cancelled"]),
                 use_container_width=True)
    st.markdown("**Classification report**")
    report = classification_report(y_test, test_pred, labels=[0, 1],
                                   target_names=["Not cancelled", "Cancelled"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    st.caption("Metrics are calculated on a stratified held-out test split. The training pipeline uses balanced sample weights to help address class imbalance.")

st.divider()
st.caption("Order Guard · AdaBoost cancellation prediction · Educational decision-support tool")
