import streamlit as st
import pandas as pd
from predictor import predict_machine, predict_batch, REQUIRED_COLS, RENAME_MAP

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")

st.title("🔧 Predictive Maintenance System")
st.caption("Machine failure prediction using the AI4I 2020 dataset")

page = st.sidebar.radio("Choose a page", ["Single Machine", "Bulk CSV Prediction"])


# ---------------------------------------------------------
# PAGE 1: SINGLE MACHINE
# ---------------------------------------------------------
if page == "Single Machine":
    st.subheader("Enter machine readings")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Product type (L / M / H)", ["L", "M", "H"])
        air_temp = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)

    with col2:
        rpm = st.slider("Rotational speed [rpm]", 1100, 2900, 1500)
        torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
        tool_wear = st.slider("Tool wear [min]", 0, 255, 50)

    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        result = predict_machine(machine_type, air_temp, process_temp, rpm, torque, tool_wear)

        prob = result["failure_probability"]
        risk = result["risk_level"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Failure probability", f"{prob * 100:.1f}%")
        m2.metric("Risk level", risk)
        m3.metric("Alert", "Yes" if result["failure_alert"] else "No")

        st.progress(min(prob, 1.0))

        if risk == "High":
            st.error("High risk: maintenance is needed soon.")
        elif risk == "Medium":
            st.warning("Medium risk: plan an inspection.")
        else:
            st.success("Low risk: the machine looks healthy.")

        if risk != "Low":
            st.write(f"**Most likely failure type:** {result['likely_cause']}")

        st.write("**Probability of each failure type**")
        st.bar_chart(pd.Series(result["cause_probabilities"]))

        st.write("**Recommended actions**")
        for item in result["advice"]:
            st.write("- " + item)


# ---------------------------------------------------------
# PAGE 2: BULK CSV PREDICTION
# ---------------------------------------------------------
if page == "Bulk CSV Prediction":
    st.subheader("Upload a CSV file")
    st.write(
        "Required columns: Type, Air temperature [K], Process temperature [K], "
        "Rotational speed [rpm], Torque [Nm], Tool wear [min]"
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        check = data.rename(columns=RENAME_MAP)
        missing = [c for c in REQUIRED_COLS if c not in check.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            results = predict_batch(data)

            c1, c2, c3 = st.columns(3)
            c1.metric("Machines", len(results))
            c2.metric("High risk", int((results["risk_level"] == "High").sum()))
            c3.metric("Alerts", int(results["failure_alert"].sum()))

            st.bar_chart(results["risk_level"].value_counts())

            st.dataframe(results.sort_values("failure_probability", ascending=False))

            st.download_button(
                "Download results as CSV",
                results.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )