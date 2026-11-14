# PhishGuard_AIMLMiniProject26
# 🛡️ PhishGuard
### Explainable and Robust Machine Learning Framework for Phishing URL Detection

PhishGuard is an AI/ML-based phishing detection framework designed to identify potentially malicious URLs using machine learning, explain individual predictions, and provide an interactive interface for security analysis.

The project combines URL-based feature extraction, webpage intelligence, XGBoost classification, SHAP explainability, and interactive graph visualization to make phishing detection more interpretable and accessible.

---

## 🚀 Live Application

**Live Demo:** [Add your Streamlit Community Cloud URL here]

**GitHub Repository:** https://github.com/akankshabhuyan-tech/PhishGuard_AIMLMiniProject26

---

## 🎯 Project Objectives

- Detect potentially phishing and legitimate URLs using machine learning.
- Extract meaningful lexical and structural features from URLs.
- Compare URL-only and URL-plus-webpage feature models.
- Explain model predictions using SHAP.
- Visualize feature contributions through an interactive PyVis graph.
- Evaluate model performance, robustness, and potential data leakage.
- Provide an accessible web interface using Streamlit.

---

## 🧠 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Matplotlib, PyVis |
| Web Application | Streamlit |
| Model Storage | JSON |
| Development Environment | Google Colab, VS Code |
| Version Control | Git, GitHub |

---

## 📊 Dataset

**Dataset:** PhiUSIIL Phishing URL Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 967

The dataset contains URL and webpage-related characteristics used to distinguish phishing websites from legitimate websites.

- Original dataset: 235,795 records.
- After duplicate URL removal: 235,370 records.
- Duplicate records removed: 425.
- Target variable: `label`.
  - `0` – Phishing
  - `1` – Legitimate

A domain-aware train-validation-test splitting strategy was used to reduce domain overlap between the datasets.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Dataset inspection and cleaning.
- Duplicate URL identification and removal.
- Conflicting label verification.
- Domain-aware data splitting.
- Feature extraction and validation.

### 2. URL-Based Feature Extraction

An independent URL feature extractor was developed to generate 18 features, including:

- URL length and domain length.
- Number of subdomains.
- IP address usage.
- HTTPS indicator.
- Special character counts.
- URL obfuscation indicators.
- Digit and letter ratios.

### 3. Machine Learning Models

Two XGBoost-based classification approaches were developed:

**Model 1: URL-Only Model**
- Uses 18 URL-derived features.
- Supports prediction using a URL without requiring webpage content.

**Model 2: URL + Webpage Model**
- Uses 46 combined URL and webpage features.
- Incorporates webpage structure and content-related indicators.

### 4. Explainable AI

SHAP (SHapley Additive exPlanations) is used to interpret model predictions and identify the features contributing to classification decisions.

### 5. Interactive Explainability Graph

PyVis is used to generate an interactive graph connecting:
- Scanned URL.
- Important extracted features.
- SHAP contribution values.
- Final prediction.

The graph helps users explore how individual features influence a prediction.

### 6. Robustness and Leakage Analysis

Additional experiments include:
- Feature-label correlation analysis.
- Duplicate feature-vector auditing.
- Conflicting-label checks.
- Domain-aware splitting.
- Strict unseen-feature-vector evaluation.
- Feature ablation experiments.
- Model comparison.

---

## 📈 Model Performance

Results obtained on the held-out test dataset:

| Metric | URL-Only Model | URL + Webpage Model |
|---|---:|---:|
| Accuracy | 99.48% | 99.99% |
| Precision (Phishing) | 99.39% | 99.99% |
| Recall (Phishing) | 99.16% | 99.99% |
| F1-Score (Phishing) | 99.27% | 99.99% |
| ROC-AUC | 99.77% | 100.00% |

**Important:** These are experimental results on the selected dataset and evaluation split. They should not be interpreted as guaranteed real-world phishing detection performance. Dataset characteristics, feature distributions, repeated feature vectors, and potential shortcuts require consideration when assessing generalization.

---

## 🖥️ Application Features

The Streamlit application provides:

- **Home:** Project overview and navigation.
- **URL Scanner:** Enter a URL and obtain a model prediction.
- **Explainability Graph:** Interactive visualization of feature contributions.
- **Model Performance:** Evaluation metrics and model comparisons.
- **Robustness Analysis:** Results from robustness and feature-ablation experiments.
- **About:** Project information and methodology.

---

## 📁 Repository Structure

```text
PhishGuard/
│
├── app/
│   ├── app.py
│   ├── feature_extractor.py
│   └── requirements.txt
│
├── artifacts/
│   ├── deployment_model.json
│   ├── url_features.json
│   └── test_metrics.json
│
├── explainability/
│   ├── deployment_shap_bar.png
│   ├── deployment_shap_summary.png
│   ├── model_comparison.csv
│   ├── deployment_robustness_results.csv
│   └── ...
│
├── notebooks/
│   └── PhishGuard_Complete_Analysis.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Running the Application Locally

### Prerequisites
- Python 3.11 or a compatible supported version.
- Git.
- VS Code or another Python IDE.

### Step 1: Clone the repository

```bash
git clone https://github.com/akankshabhuyan-tech/PhishGuard_AIMLMiniProject26.git
cd PhishGuard_AIMLMiniProject26
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```bash
python -m streamlit run app/app.py
```

### Step 4: Open the application

Visit:

```text
http://localhost:8501
```

---

## 🔍 Limitations and Responsible Use

- Model predictions are probabilistic and should not be treated as definitive security guarantees.
- Webpage-based predictions depend on the availability and quality of webpage features.
- Website content may change over time.
- The experimental dataset may not represent all real-world phishing techniques.
- Users should avoid entering credentials or sensitive information into suspicious websites.
- PhishGuard is an academic research and demonstration project, not a replacement for professional cybersecurity solutions.

---

## 🔮 Future Enhancements

- Real-time webpage feature extraction with improved security controls.
- Integration of additional threat intelligence sources.
- Evaluation on independently collected phishing datasets.
- Improved generalization to newly emerging phishing techniques.
- Expanded explainability and interactive security analytics.
- Continuous model monitoring and updates.

---

## 👩‍💻 Project Information

**Project Name:** PhishGuard – Explainable and Robust Machine Learning Framework for Phishing URL Detection

**Domain:** Artificial Intelligence, Machine Learning and Cybersecurity

**Institution:** Amrita Vishwa Vidyapeetham, Bengaluru

**Project Type:** Academic Mini Project

---

## 📜 Disclaimer

This project is developed for educational and research purposes. The developers are not responsible for decisions or actions taken solely on the basis of its predictions.
