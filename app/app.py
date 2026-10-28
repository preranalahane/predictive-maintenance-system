import streamlit as st
import pandas as pd
from predictor import (
    predict_machine, predict_batch,
    REQUIRED_COLS, RENAME_MAP, QUALITY_MAP, MACHINE_MAP
)

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")

# ---------------------------------------------------------
# Colours: (background, text colour)
# ---------------------------------------------------------
RISK_STYLE = {
    "High": ("#D32F2F", "#FFFFFF"),      # red
    "Medium": ("#FBC02D", "#212121"),    # yellow
    "Low": ("#2E7D32", "#FFFFFF"),       # green
}


def risk_card(risk, prob):
    bg, fg = RISK_STYLE[risk]
    st.markdown(
        f"<div style='background-color:{bg}; padding:20px; border-radius:12px; text-align:center;'>"
        f"<div style='color:{fg}; font-size:22px; font-weight:bold;'>{risk} Risk</div>"
        f"<div style='color:{fg}; font-size:48px; font-weight:bold;'>{prob * 100:.1f}%</div>"
        f"<div style='color:{fg}; font-size:16px;'>Failure probability</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def prob_bar(risk, prob):
    bg, fg = RISK_STYLE[risk]
    width = min(max(prob, 0.0), 1.0) * 100
    st.markdown(
        f"<div style='background-color:#E0E0E0; border-radius:8px; height:14px; margin:12px 0;'>"
        f"<div style='background-color:{bg}; width:{width:.1f}%; height:14px; border-radius:8px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def count_card(risk, count):
    bg, fg = RISK_STYLE[risk]
    st.markdown(
        f"<div style='background-color:{bg}; padding:12px; border-radius:10px; text-align:center;'>"
        f"<div style='color:{fg}; font-size:16px; font-weight:bold;'>{risk} risk</div>"
        f"<div style='color:{fg}; font-size:32px; font-weight:bold;'>{count}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def highlight_row(row):
    bg, fg = RISK_STYLE[row["risk_level"]]
    return [
        f"background-color: {bg}; color: {fg}; font-weight: bold"
        if col in ("failure_probability", "risk_level") else ""
        for col in row.index
    ]


# ---------------------------------------------------------
# Title and sidebar
# ---------------------------------------------------------
st.title("🔧 Predictive Maintenance System")
st.caption("Milling and turning machine failure prediction (augmented AI4I 2020 dataset)")

page = st.sidebar.radio("Choose a page", ["Single Machine", "Bulk CSV Prediction"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Classification**")

for level, label in [
    ("Low", "Below 30%"),
    ("Medium", "30% to below 60%"),
    ("High", "60% and above"),
]:
    bg = RISK_STYLE[level][0]
    st.sidebar.markdown(
        f"<div style='margin-bottom:6px;'>"
        f"<span style='display:inline-block; width:12px; height:12px; "
        f"background-color:{bg}; border-radius:3px; margin-right:8px;'></span>"
        f"<b>{level}</b>: {label}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# PAGE 1: SINGLE MACHINE
# ---------------------------------------------------------
if page == "Single Machine":
    st.subheader("Enter machine readings")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine type", ["Milling", "Turning"])
        quality_type = st.selectbox("Product quality type (L / M / H)", ["L", "M", "H"])
        air_temp = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)

    with col2:
        rpm = st.slider("Rotational speed [rpm]", 1100, 2900, 1500)
        torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
        cutting_force = st.slider("Cutting force [N]", 200, 6000, 2500, 10)
        flank_wear = st.slider("Tool flank wear VB [mm]", 0.000, 0.450, 0.100, 0.005)

    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        result = predict_machine(quality_type, machine_type, air_temp, process_temp,
                                 rpm, torque, cutting_force, flank_wear)

        prob = result["failure_probability"]
        risk = result["risk_level"]

        st.divider()

        left, right = st.columns([2, 1])

        with left:
            risk_card(risk, prob)
            prob_bar(risk, prob)

        with right:
            st.metric("Alert", "Yes" if result["failure_alert"] else "No")
            if risk != "Low":
                st.metric("Most likely failure type", result["likely_cause"])

        if risk == "High":
            st.error("High risk: maintenance is needed soon.")
        elif risk == "Medium":
            st.warning("Medium risk: plan an inspection.")
        else:
            st.success("Low risk: the machine looks healthy.")

        if risk != "Low":
            st.write("**Probability of each failure type**")
            st.bar_chart(pd.Series(result["cause_probabilities"]), color=RISK_STYLE[risk][0])

        st.write("**Recommended actions**")
        for item in result["advice"]:
            st.write("- " + item)

        with st.expander("Calculated features"):
            calc = result["calculated"]
            st.write(f"Temperature difference: {calc['temp_diff']} K")
            st.write(f"Power: {calc['power']} W")
            st.write(f"Wear x force: {calc['wear_force']}")


# ---------------------------------------------------------
# PAGE 2: BULK CSV PREDICTION
# ---------------------------------------------------------
if page == "Bulk CSV Prediction":
    st.subheader("Upload a CSV file")
    st.write(
        "Required columns: Quality Type, Machine Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], Torque [Nm], "
        "Cutting force [N], Tool flank wear VB [mm]"
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        check = data.rename(columns=RENAME_MAP)
        missing = [c for c in REQUIRED_COLS if c not in check.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            bad_quality = set(check["quality_type"].dropna().unique()) - set(QUALITY_MAP)
            bad_machine = set(check["machine_type"].dropna().unique()) - set(MACHINE_MAP)

            if bad_quality or bad_machine:
                st.error(
                    f"Unknown values found. Quality Type must be L, M or H. "
                    f"Machine Type must be Milling or Turning. "
                    f"Found: {bad_quality | bad_machine}"
                )
            elif check[REQUIRED_COLS].isnull().any().any():
                st.error("The file has empty values in the required columns.")
            else:
                results = predict_batch(data)

                n_high = int((results["risk_level"] == "High").sum())
                n_medium = int((results["risk_level"] == "Medium").sum())
                n_low = int((results["risk_level"] == "Low").sum())

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Machines", len(results))
                c2.metric("Alerts", int(results["failure_alert"].sum()))
                with c3:
                    count_card("High", n_high)
                with c4:
                    count_card("Medium", n_medium)
                with c5:
                    count_card("Low", n_low)

                st.write("")
                sorted_results = results.sort_values("failure_probability", ascending=False)

                show_n = min(500, len(sorted_results))
                st.write(f"Showing the {show_n} highest-risk machines (download the file for all rows).")

                styled = sorted_results.head(show_n).style.apply(highlight_row, axis=1)
                st.dataframe(styled)

                st.download_button(
                    "Download results as CSV",
                    results.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )