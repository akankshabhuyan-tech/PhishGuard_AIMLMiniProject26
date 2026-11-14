
import os
import sys
import json
import html
import ipaddress
from urllib.parse import urlsplit

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier
from pyvis.network import Network

# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explainability")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from feature_extractor import extract_url_features


# ---------------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="PhishGuard | Phishing URL Detection",
    page_icon="🛡️",
    layout="wide"
)

# Light-themed styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7FAFF;
    }

    h1, h2, h3 {
        color: #243B64;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE6F1;
        padding: 15px;
        border-radius: 12px;
    }

    .main-card {
        background-color: #FFFFFF;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #DCE6F1;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# 3. LOAD MODEL AND ARTIFACTS
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model_path = os.path.join(
        ARTIFACTS_DIR,
        "deployment_model.json"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = XGBClassifier()
    model.load_model(model_path)

    return model


@st.cache_data
def load_feature_names():
    feature_path = os.path.join(
        ARTIFACTS_DIR,
        "url_features.json"
    )

    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}"
        )

    with open(feature_path, "r", encoding="utf-8") as file:
        features = json.load(file)

    # Support either a direct list or a dictionary containing the list.
    if isinstance(features, dict):
        for key in ["features", "feature_names", "url_features"]:
            if key in features:
                features = features[key]
                break

    if not isinstance(features, list):
        raise ValueError(
            "url_features.json must contain a list of feature names."
        )

    return features


@st.cache_data
def load_test_metrics():
    metrics_path = os.path.join(
        ARTIFACTS_DIR,
        "test_metrics.json"
    )

    if not os.path.exists(metrics_path):
        return {}

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    feature_names = load_feature_names()
    test_metrics = load_test_metrics()

except Exception as error:
    st.error("PhishGuard could not load its model or artifacts.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# 4. URL VALIDATION
# ---------------------------------------------------------

def validate_url(url):
    url = url.strip()

    if not url:
        raise ValueError("Please enter a URL.")

    if any(ord(char) < 32 for char in url):
        raise ValueError("The URL contains invalid control characters.")

    if any(char.isspace() for char in url):
        raise ValueError("The URL must not contain spaces.")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname

        if not hostname:
            raise ValueError("The URL does not contain a valid hostname.")

        # Reject malformed ports.
        _ = parsed.port

    except ValueError as error:
        raise ValueError(f"Invalid URL: {error}")

    return url


# ---------------------------------------------------------
# 5. EXTRACT FEATURES AND PREDICT
# ---------------------------------------------------------

def predict_url(url):
    extracted = extract_url_features(url)

    if isinstance(extracted, pd.DataFrame):
        input_df = extracted.copy()

    elif isinstance(extracted, pd.Series):
        input_df = extracted.to_frame().T

    elif isinstance(extracted, dict):
        input_df = pd.DataFrame([extracted])

    else:
        raise TypeError(
            "The feature extractor must return a dictionary, "
            "Series, or DataFrame."
        )

    # Ensure the model receives features in the correct order.
    missing_features = [
        feature for feature in feature_names
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The feature extractor did not produce these required "
            f"features: {missing_features}"
        )

    input_df = input_df[feature_names].copy()
    input_df = input_df.apply(pd.to_numeric, errors="coerce")

    if input_df.isnull().any().any():
        bad_features = input_df.columns[
            input_df.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing or non-numeric feature values: {bad_features}"
        )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    phishing_probability = float(probabilities[0])
    legitimate_probability = float(probabilities[1])

    return (
        prediction,
        phishing_probability,
        legitimate_probability,
        input_df
    )


# ---------------------------------------------------------
# 6. GET SHAP VALUES FOR PHISHING CLASS
# ---------------------------------------------------------

def get_phishing_shap_values(input_df):
    """
    The model uses:
        Class 0 = Phishing
        Class 1 = Legitimate

    For binary XGBoost, TreeExplainer commonly returns SHAP
    values for class 1. We reverse their sign to explain
    class 0 (phishing).
    """

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(input_df)

        if isinstance(raw_values, list):
            # For a two-class list, index 0 represents class 0.
            if len(raw_values) == 2:
                phishing_values = np.asarray(raw_values[0])[0]
            else:
                phishing_values = np.asarray(raw_values[0])[0]

        else:
            values = np.asarray(raw_values)

            if values.ndim == 3:
                # Shape: samples x features x classes
                if values.shape[-1] == 2:
                    phishing_values = values[0, :, 0]
                else:
                    phishing_values = values[0, :, 0]

            elif values.ndim == 2:
                # Shape: samples x features.
                # Binary XGBoost SHAP values generally explain class 1.
                phishing_values = -values[0]

            elif values.ndim == 1:
                phishing_values = -values

            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: {values.shape}"
                )

        phishing_values = np.asarray(
            phishing_values,
            dtype=float
        ).flatten()

        if len(phishing_values) != len(feature_names):
            raise ValueError(
                "SHAP output does not match the model feature count."
            )

        return explainer, phishing_values

    except Exception as error:
        st.warning(f"SHAP explanation could not be generated: {error}")
        return None, None


# ---------------------------------------------------------
# 7. DISPLAY SHAP EXPLANATION AND TOP 10 FEATURES
# ---------------------------------------------------------

def show_shap_explanation(input_df):
    st.subheader("🔍 Explainable AI — SHAP Analysis")

    explainer, shap_values = get_phishing_shap_values(input_df)

    if shap_values is None:
        st.info("SHAP explanation is currently unavailable.")
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": input_df.iloc[0].values,
        "SHAP Contribution": shap_values,
        "Absolute Importance": np.abs(shap_values)
    })

    importance_df = importance_df.sort_values(
        by="Absolute Importance",
        ascending=False
    ).reset_index(drop=True)

    st.write(
        "The table ranks the features by the magnitude of their "
        "contribution to the phishing-class explanation."
    )

    st.dataframe(
        importance_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        **How to interpret the SHAP values**

        - Positive SHAP contribution: pushes the prediction towards **Phishing**.
        - Negative SHAP contribution: pushes the prediction towards **Legitimate**.
        - Larger absolute values indicate a stronger contribution.
        """
    )

    # SHAP bar chart
    st.markdown("#### Top 10 feature contributions")

    plot_df = importance_df.head(10).sort_values(
        "SHAP Contribution"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_colors = [
        "#E88C8C" if value > 0 else "#78AEDD"
        for value in plot_df["SHAP Contribution"]
    ]

    ax.barh(
        plot_df["Feature"],
        plot_df["SHAP Contribution"],
        color=bar_colors
    )

    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("SHAP contribution towards phishing")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 SHAP Feature Contributions")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    return shap_values


# ---------------------------------------------------------
# 8. INTERACTIVE PYVIS EXPLAINABILITY GRAPH
# ---------------------------------------------------------

def create_explainability_graph(
    url,
    prediction,
    input_df,
    shap_values
):
    st.subheader("🕸️ Interactive Explainability Graph")

    st.write(
        "Explore how the scanned URL connects to its extracted "
        "features and how those features influence the prediction. "
        "Hover over the nodes and edges for more information."
    )

    if shap_values is None:
        st.info("The graph requires SHAP values and cannot be displayed.")
        return

    try:
        shap_values = np.asarray(
            shap_values,
            dtype=float
        ).flatten()

        if len(shap_values) != len(feature_names):
            st.warning("SHAP values do not match the feature list.")
            return

        top_indices = np.argsort(
            np.abs(shap_values)
        )[-10:][::-1]

        net = Network(
            height="650px",
            width="100%",
            directed=True,
            bgcolor="#FFFFFF",
            font_color="#263238",
            cdn_resources="in_line"
        )

        net.add_node(
            "url",
            label="Scanned URL",
            title=html.escape(url),
            shape="box",
            color="#B3D9FF",
            size=30,
            font={"size": 20}
        )

        prediction_label = (
            "PHISHING" if prediction == 0 else "LEGITIMATE"
        )

        prediction_color = (
            "#FFB3B3" if prediction == 0 else "#B8E6C1"
        )

        net.add_node(
            "prediction",
            label=prediction_label,
            title="Final XGBoost model prediction",
            shape="box",
            color=prediction_color,
            size=32,
            font={"size": 20}
        )

        for idx in top_indices:
            feature = feature_names[idx]
            feature_value = input_df.iloc[0][feature]
            contribution = float(shap_values[idx])

            if contribution >= 0:
                feature_color = "#FFCCCC"
                direction = "Pushes towards phishing"
                edge_color = "#E57373"
            else:
                feature_color = "#C9E8FF"
                direction = "Pushes towards legitimate"
                edge_color = "#64A5D8"

            feature_id = f"feature_{idx}"

            tooltip = (
                f"Feature: {html.escape(str(feature))}<br>"
                f"Value: {html.escape(str(feature_value))}<br>"
                f"SHAP contribution: {contribution:.5f}<br>"
                f"Interpretation: {direction}"
            )

            net.add_node(
                feature_id,
                label=str(feature),
                title=tooltip,
                shape="dot",
                color=feature_color,
                size=20 + min(abs(contribution) * 100, 20),
                font={"size": 14}
            )

            net.add_edge(
                "url",
                feature_id,
                color="#A0AEC0",
                arrows="to"
            )

            net.add_edge(
                feature_id,
                "prediction",
                color=edge_color,
                width=1 + min(abs(contribution) * 5, 5),
                arrows="to",
                title=f"SHAP contribution: {contribution:.5f}"
            )

        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.25,
            spring_length=180,
            spring_strength=0.04,
            damping=0.09
        )

        graph_html = net.generate_html()

        components.html(
            graph_html,
            height=680,
            scrolling=True
        )

        st.caption(
            "The graph displays the 10 features with the largest "
            "absolute SHAP contributions. Red indicates a push "
            "towards phishing; blue indicates a push towards legitimate."
        )

    except Exception as error:
        st.error("Could not generate the interactive graph.")
        st.exception(error)


# ---------------------------------------------------------
# 9. SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("🛡️ PhishGuard")

st.sidebar.write(
    "Explainable and Robust Machine Learning "
    "Framework for Phishing URL Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "URL Scanner",
        "Model Performance",
        "Robustness Analysis",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.caption("PhishGuard | Machine Learning Project")


# ---------------------------------------------------------
# 10. HOME PAGE
# ---------------------------------------------------------

if page == "Home":

    st.title("🛡️ Welcome to PhishGuard")

    st.markdown(
        """
        ### Explainable and Robust Machine Learning Framework
        ### for Phishing URL Detection

        PhishGuard uses machine learning to analyse URL characteristics
        and identify potentially phishing websites.

        The system also provides explainable AI insights so users can
        understand which URL features influenced a prediction.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="main-card"><h3>🔎 URL Scanning</h3>'
            '<p>Analyse a URL and receive a phishing risk prediction.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="main-card"><h3>🧠 Explainable AI</h3>'
            '<p>Understand feature contributions using SHAP.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div class="main-card"><h3>🕸️ Interactive Graph</h3>'
            '<p>Explore relationships between URLs, features, and predictions.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.info(
        "Navigate to URL Scanner from the sidebar to analyse a URL."
    )


# ---------------------------------------------------------
# 11. URL SCANNER PAGE
# ---------------------------------------------------------

elif page == "URL Scanner":

    st.title("🔎 Phishing URL Scanner")

    st.write(
        "Enter a website URL below. PhishGuard will extract URL "
        "features and analyse them using the trained XGBoost model."
    )

    url_input = st.text_input(
        "Enter URL",
        placeholder="https://example.com"
    )

    scan_button = st.button(
        "Scan URL",
        type="primary",
        use_container_width=True
    )

    if scan_button:

        try:
            clean_url = validate_url(url_input)

            with st.spinner("Extracting URL features and analysing..."):

                (
                    prediction,
                    phishing_probability,
                    legitimate_probability,
                    input_df
                ) = predict_url(clean_url)

            st.divider()
            st.subheader("📊 Scan Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 0:
                    st.error("⚠️ PHISHING URL DETECTED")

                else:
                    st.success("✅ LEGITIMATE URL PREDICTED")

            with result_col2:
                st.metric(
                    "Phishing Probability",
                    f"{phishing_probability * 100:.2f}%"
                )

            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:
                st.metric(
                    "Phishing",
                    f"{phishing_probability * 100:.2f}%"
                )

            with probability_col2:
                st.metric(
                    "Legitimate",
                    f"{legitimate_probability * 100:.2f}%"
                )

            st.progress(phishing_probability)

            st.caption(
                "This is a machine-learning prediction, not a guarantee "
                "that a website is safe or malicious. Avoid entering "
                "passwords or sensitive information on suspicious sites."
            )

            st.divider()

            # Extracted URL features
            with st.expander("View extracted URL features"):
                st.dataframe(
                    input_df.T.rename(
                        columns={0: "Extracted Value"}
                    ),
                    use_container_width=True
                )

            st.divider()

            # SHAP explanation
            shap_values = show_shap_explanation(input_df)

            st.divider()

            # Interactive graph
            create_explainability_graph(
                clean_url,
                prediction,
                input_df,
                shap_values
            )

        except Exception as error:
            st.error("The URL could not be analysed.")
            st.exception(error)


# ---------------------------------------------------------
# 12. MODEL PERFORMANCE PAGE
# ---------------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "The following metrics describe the performance of the "
        "trained URL-only XGBoost deployment model on the held-out test set."
    )

    if test_metrics:

        # Display common metrics if they exist in the JSON.
        metric_aliases = {
            "Accuracy": ["accuracy", "Accuracy"],
            "Precision": ["precision", "Precision"],
            "Recall": ["recall", "Recall"],
            "F1 Score": ["f1", "f1_score", "F1", "F1 Score"],
            "ROC-AUC": ["roc_auc", "roc-auc", "ROC-AUC", "auc"]
        }

        found_metrics = {}

        # Search both top-level keys and nested common containers.
        metric_sources = [test_metrics]

        for nested_key in ["metrics", "test_metrics", "results"]:
            if isinstance(test_metrics.get(nested_key), dict):
                metric_sources.append(test_metrics[nested_key])

        for display_name, possible_keys in metric_aliases.items():
            for source in metric_sources:
                for key in possible_keys:
                    if key in source and isinstance(
                        source[key], (int, float)
                    ):
                        found_metrics[display_name] = source[key]
                        break

                if display_name in found_metrics:
                    break

        if found_metrics:
            metric_cols = st.columns(len(found_metrics))

            for column, (name, value) in zip(
                metric_cols,
                found_metrics.items()
            ):
                with column:
                    st.metric(name, f"{value:.4f}")

        else:
            st.info(
                "The test_metrics.json file was loaded, but its metric "
                "keys do not match the display names expected here."
            )

        with st.expander("View raw test metrics"):
            st.json(test_metrics)

    else:
        st.warning("test_metrics.json was not found or is empty.")

    # Optional model comparison CSV
    comparison_path = os.path.join(
        EXPLAIN_DIR,
        "model_comparison.csv"
    )

    if os.path.exists(comparison_path):

        st.divider()
        st.subheader("URL Model vs URL + Webpage Model")

        comparison_df = pd.read_csv(comparison_path)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The URL + webpage results are reported from the offline "
            "evaluation. The live scanner on this page currently uses "
            "the URL-only deployment model."
        )


# ---------------------------------------------------------
# 13. ROBUSTNESS ANALYSIS PAGE
# ---------------------------------------------------------

elif page == "Robustness Analysis":

    st.title("🧪 Robustness Analysis")

    st.write(
        "This section displays the saved robustness evaluation "
        "results for the URL-only deployment model."
    )

    robustness_path = os.path.join(
        EXPLAIN_DIR,
        "deployment_robustness_results.csv"
    )

    if os.path.exists(robustness_path):

        robustness_df = pd.read_csv(robustness_path)

        st.subheader("Robustness Results")

        st.dataframe(
            robustness_df,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = robustness_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_metric = st.selectbox(
                "Select metric to visualise",
                numeric_columns
            )

            chart_df = robustness_df[
                [robustness_df.columns[0], selected_metric]
            ].copy()

            chart_df = chart_df.set_index(
                robustness_df.columns[0]
            )

            st.line_chart(chart_df)

        else:
            st.info(
                "No numeric columns were found for charting."
            )

    else:
        st.warning(
            "deployment_robustness_results.csv was not found "
            "inside the explainability folder."
        )

    robustness_image = os.path.join(
        EXPLAIN_DIR,
        "deployment_robustness.png"
    )

    if os.path.exists(robustness_image):
        st.subheader("Saved Robustness Plot")
        st.image(
            robustness_image,
            use_container_width=True
        )


# ---------------------------------------------------------
# 14. ABOUT PAGE
# ---------------------------------------------------------

elif page == "About":

    st.title("ℹ️ About PhishGuard")

    st.markdown(
        """
        ### Project Objective

        PhishGuard aims to identify phishing URLs using machine learning
        and provide interpretable explanations for its predictions.

        ### Technologies Used

        - Python
        - XGBoost
        - SHAP
        - Streamlit
        - PyVis
        - Pandas
        - NumPy
        - Matplotlib

        ### Key Components

        1. URL feature extraction
        2. XGBoost-based phishing classification
        3. SHAP-based explainability
        4. Interactive URL-feature-prediction graph
        5. Model performance evaluation
        6. Robustness analysis

        ### Important Limitation

        The live scanner currently analyses URL-derived features.
        It does not independently inspect the live webpage or guarantee
        that a predicted legitimate URL is safe.
        """
    )

    st.info(
        "PhishGuard is an academic machine-learning project "
        "and should not be treated as a replacement for browser security."
    )


# ---------------------------------------------------------
# 15. FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "PhishGuard | Explainable and Robust Machine Learning "
    "Framework for Phishing URL Detection"
)
