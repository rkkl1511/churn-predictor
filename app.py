import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return pipe, meta, explainer


pipe, meta, explainer = load_artifacts()
opts = meta["cat_options"]
rng = meta["num_ranges"]

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details to estimate their churn risk and see "
    "which factors drive the prediction."
)

# ---------- Inputs ----------
st.sidebar.header("Customer details")

st.sidebar.subheader("Account")
tenure = st.sidebar.slider("Tenure (months)", 0, int(rng["Tenure_Months"][1]),
                           int(rng["Tenure_Months"][2]))
monthly = st.sidebar.slider("Monthly charges ($)", 15.0, 125.0,
                            float(round(rng["Monthly_Charges"][2], 2)), step=0.5)
contract = st.sidebar.selectbox("Contract", opts["Contract"])
paperless = st.sidebar.selectbox(
    "Paperless billing", opts["Paperless_Billing"])
payment = st.sidebar.selectbox("Payment method", opts["Payment_Method"])

st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", opts["Gender"])
senior = st.sidebar.selectbox("Senior citizen", opts["Senior_Citizen"])
partner = st.sidebar.selectbox("Partner", opts["Partner"])
dependents = st.sidebar.selectbox("Dependents", opts["Dependents"])

st.sidebar.subheader("Services")
phone = st.sidebar.selectbox("Phone service", opts["Phone_Service"])
if phone == "No":
    multiple_lines = "No phone service"
else:
    multiple_lines = st.sidebar.selectbox("Multiple lines", ["No", "Yes"])

internet = st.sidebar.selectbox("Internet service", opts["Internet_Service"])
addon_cols = ["Online_Security", "Online_Backup", "Device_Protection",
              "Tech_Support", "Streaming_TV", "Streaming_Movies"]
addons = {}
for col in addon_cols:
    if internet == "No":
        addons[col] = "No internet service"
    else:
        addons[col] = st.sidebar.selectbox(
            col.replace("_", " "), ["No", "Yes"])

# ---------- Build the input row ----------
row = {
    "Gender": gender,
    "Senior_Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure_Months": tenure,
    "Phone_Service": phone,
    "Multiple_Lines": multiple_lines,
    "Internet_Service": internet,
    **addons,
    "Contract": contract,
    "Paperless_Billing": paperless,
    "Payment_Method": payment,
    "Monthly_Charges": monthly,
    "Total_Charges": tenure * monthly,  # estimated from tenure and monthly charges
}
X_new = pd.DataFrame([row])[meta["feature_order"]]

# ---------- Predict ----------
score = float(pipe.predict_proba(X_new)[0, 1])

if score < 0.35:
    level, color = "Low risk", "green"
elif score < 0.65:
    level, color = "Medium risk", "orange"
else:
    level, color = "High risk", "red"

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    st.metric("Churn risk score", f"{score:.0%}")
    st.markdown(f"### :{color}[{level}]")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption(
        "The model was trained with churners weighted more heavily, so this is "
        "a risk score rather than a calibrated probability."
    )
    m = meta["metrics"]
    st.markdown("**Model performance (test set)**")
    st.write(
        f"ROC-AUC: {m['roc_auc']}  \nRecall: {m['recall']}  \nPrecision: {m['precision']}")

# ---------- Explain ----------
with col2:
    st.subheader("Why this prediction?")
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X_new)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = [n.replace("num__", "").replace("cat__", "")
             for n in prep.get_feature_names_out()]
    shap_values = explainer(pd.DataFrame(Xt, columns=names))

    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption(
        "Red bars push the customer toward churn, blue bars push away. "
        "Numeric values (tenure, charges) are shown standardized. "
        "Total charges are estimated as tenure × monthly charges."
    )