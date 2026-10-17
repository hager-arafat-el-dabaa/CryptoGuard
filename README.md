# CryptoGuard: Semi-Supervised AML & Transaction Anomaly Detection

**Detecting illicit Bitcoin transactions on the Elliptic dataset using a temporal train/test split, a temporal graph-degree feature, SMOTE, gradient-boosted ensembles, pseudo-labeling of unlabeled transactions, and an Isolation Forest baseline, with a Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-models%20%26%20metrics-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-supervised-green) ![LightGBM](https://img.shields.io/badge/LightGBM-supervised-brightgreen) ![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple) ![Dataset](https://img.shields.io/badge/Dataset-Elliptic-lightgrey)

🔗 **Live Dashboard:** [CryptoGuard Dashboard](https://cryptoguard-dashboard-h7qybj5g2jbmjpkgvdvd7v.streamlit.app/dataset)

---

##  Table of Contents

1. [Overview](#-overview)
2. [Results at a Glance](#-results-at-a-glance)
3. [Dataset](#-dataset)
4. [Pipeline](#-pipeline)
5. [Methodology in Detail](#-methodology-in-detail)
6. [Experimental Setup](#-experimental-setup)
7. [Results](#-results)
8. [Analysis](#-analysis)
9. [Dashboard](#-dashboard)
10. [Repository Structure](#-repository-structure)
11. [Technologies](#-technologies)
12. [Installation and Local Execution](#-installation-and-local-execution)
13. [Limitations](#-limitations)
14. [Future Improvements](#-future-improvements)
15. [Author](#-author)

---

##  Overview

Bitcoin transactions are pseudonymous, making it difficult to distinguish illicit activity from legitimate transactions. Manually reviewing every transaction is impractical. CryptoGuard uses machine learning to identify transactions that are likely to be illicit and help prioritize further investigation.

The project uses the **Elliptic Bitcoin dataset**, containing 203,769 transactions, 234,355 edges, and 49 time steps.

Three learning approaches are explored:

| Approach | Implementation |
|---|---|
|  **Supervised Learning** | Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and a Soft Voting Ensemble combining Random Forest, XGBoost, and LightGBM |
|  **Semi-Supervised Learning** | Pseudo-labeling high-confidence predictions for unlabeled transactions from training time steps, followed by model retraining |
|  **Unsupervised Learning** | Isolation Forest trained on unlabeled transactions from the training period and evaluated against the illicit labels in the test period |

> [!NOTE]
> The semi-supervised experiment uses pseudo-labeling, while the unsupervised experiment uses a single Isolation Forest model. Other semi-supervised and unsupervised methods are not implemented.

All approaches are evaluated on the same chronological test period, time steps 35–49, using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC. The illicit class is treated as the positive class.

---

##  Results at a Glance

**Test set:** Time steps 35–49, containing 16,670 labeled transactions, including 1,083 illicit transactions. The classification threshold is fixed at 0.5.

| Highlight | Model | Result |
|---|---|---|
| Best supervised F1-score | Soft Voting Ensemble | **0.8173** |
| Best supervised PR-AUC | LightGBM | **0.8190** |
| Best supervised ROC-AUC | LightGBM | **0.9488** |
| Highest F1-score among evaluated models | Semi-Supervised Random Forest | **0.8221** |
| Highest Recall | Logistic Regression | **0.8550** |
| Unsupervised baseline | Isolation Forest | F1-score: **0.0363** |

### F1-score comparison

```text
Semi Random Forest   0.8221
Soft Voting Ensemble 0.8173
LightGBM             0.8149
XGBoost              0.8107
Random Forest        0.8066
Decision Tree        0.5430
Logistic Regression  0.3191
Isolation Forest     0.0363
```

---

##  Dataset

**Source:** Elliptic Bitcoin dataset.

The dataset consists of three files:

| File | Description | Shape |
|---|---|---|
| `elliptic_txs_features.csv` | Transaction IDs, time steps, and transaction features | 203,769 × 167 |
| `elliptic_txs_classes.csv` | Transaction IDs and class labels | 203,769 × 2 |
| `elliptic_txs_edgelist.csv` | Transaction relationships (`txId1`, `txId2`) | 234,355 × 2 |

The features file contains a transaction ID, a time step, and 165 original features. One engineered graph-degree feature is added, resulting in **166 model features**. The time step itself is excluded from model training.

### Class distribution

| Class | Meaning | Count | Percentage of all transactions |
|---|---|---:|---:|
| `1` | Illicit | 4,545 | 2.23% |
| `2` | Licit | 42,019 | 20.62% |
| `unknown` | Unlabeled | 157,205 | 77.15% |

Only 46,564 transactions have known labels. Among these labeled transactions, approximately 9.76% are illicit, making metrics such as Precision, Recall, F1-score, and PR-AUC particularly important.

Unlabeled transactions are kept separate from the supervised training data and are used in the semi-supervised and unsupervised experiments.

### Data integrity checks

The notebook includes checks for:

- Missing transaction IDs between the features and classes files.
- Edge endpoints absent from the features file.
- Missing values in the dataset files.
- Duplicate rows in the dataset files.

### Train/test composition

| Split | Time steps | Labeled transactions | Licit | Illicit | Illicit percentage |
|---|---|---:|---:|---:|---:|
| Train | 1–34 | 29,894 | 26,432 | 3,462 | 11.58% |
| Test | 35–49 | 16,670 | 15,587 | 1,083 | 6.50% |

---

##  Pipeline

1. Load the three Elliptic dataset files.
2. Perform data integrity checks.
3. Merge transaction features and class labels.
4. Engineer a temporal graph-degree feature.
5. Separate labeled and unlabeled transactions.
6. Split labeled transactions chronologically into training and test periods.
7. Scale features using `StandardScaler`, fitted on the training set only.
8. Apply SMOTE to the supervised training data.
9. Train and evaluate supervised models.
10. Generate pseudo-labels for high-confidence unlabeled transactions from training time steps.
11. Retrain models using real and pseudo-labeled data.
12. Train Isolation Forest on unlabeled transactions from the training period.
13. Evaluate the approaches on the same chronological test set.

---

## 🔬 Methodology in Detail

### 1. Temporal graph-degree feature

The degree of a transaction represents its number of connections in the edge list.

To reduce future-information leakage, an edge is counted only when both transactions involved are available at the relevant time step. Source-side and target-side connection counts are computed separately and combined into one degree feature.

Degree statistics across all transactions:

- Mean: 2.30
- Standard deviation: 4.33
- Minimum: 1
- Median: 2
- Maximum: 473

### 2. Chronological train/test split

| Split | Time steps |
|---|---|
| Training | 1–34 |
| Testing | 35–49 |

No validation set is used. The test set is not used for threshold tuning, and the classification threshold is fixed at 0.5.

### 3. Feature scaling and class balancing

- `StandardScaler` is fitted on the training data only.
- The fitted scaler is applied to the test data and unlabeled transactions.
- `SMOTE(random_state=42)` is applied to the supervised training set only.

The original training distribution of 26,432 licit and 3,462 illicit transactions is balanced to 26,432 samples per class.

### 4. Supervised models

All supervised models are trained using the SMOTE-balanced training set.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | Default settings |
| Random Forest | `n_estimators=200` |
| XGBoost | `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`, `eval_metric="logloss"` |
| LightGBM | `n_estimators=200`, `learning_rate=0.1`, `max_depth=6` |
| Soft Voting Ensemble | Average of illicit-class probabilities from Random Forest, XGBoost, and LightGBM |

### 5. Semi-supervised learning: pseudo-labeling

The semi-supervised experiment uses unlabeled transactions from time steps 1–34 only.

1. Select the 106,371 unlabeled transactions from training time steps.
2. Predict their probabilities using the supervised ensemble of Random Forest, XGBoost, and LightGBM.
3. Keep high-confidence predictions using the following thresholds:

| Probability | Assigned label | Count |
|---|---|---:|
| ≥ 0.90 | Illicit | 2,002 |
| ≤ 0.10 | Licit | 77,320 |
| Between 0.10 and 0.90 | Discarded | 27,049 |

4. Randomly sample 2,002 high-confidence licit transactions to balance the pseudo-labeled data.
5. Combine the 4,004 pseudo-labeled samples with the 29,894 real training samples, resulting in 33,898 rows.
6. Retrain Random Forest, XGBoost, and LightGBM, then construct the semi-supervised Soft Voting Ensemble.

> [!IMPORTANT]
> The semi-supervised models use the scaled real training data plus pseudo-labeled samples without SMOTE. The supervised models use SMOTE-balanced training data. Therefore, the comparison does not isolate the effect of pseudo-labeling alone.

### 6. Unsupervised anomaly detection

The unsupervised experiment uses:

`IsolationForest(n_estimators=200, contamination=0.05, random_state=42)`

The model is fitted on scaled unlabeled transactions from time steps 1–34 without using class labels.

For evaluation:

- Predicted anomalies are treated as illicit predictions when calculating Accuracy, Precision, Recall, and F1-score.
- The negated `decision_function` is used to calculate ROC-AUC and PR-AUC, with higher scores indicating greater anomaly levels.

An anomaly does not necessarily represent illicit activity, so Isolation Forest is treated as a complementary baseline.

---

##  Experimental Setup

| Item | Setting |
|---|---|
| Training period | Time steps 1–34 |
| Test period | Time steps 35–49 |
| Validation set | None |
| Classification threshold | 0.5 |
| Positive class | Illicit |
| Random seed | 42 |
| Evaluation metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC |
| Model features | 165 original features + graph degree = 166 |

---

##  Results

All results below are from the notebook's test-set outputs for time steps 35–49.

### Supervised models

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Soft Voting Ensemble** | 0.9789 | **0.9368** | 0.7248 | **0.8173** | 0.9397 | 0.8086 |
| LightGBM | 0.9785 | 0.9260 | 0.7276 | 0.8149 | **0.9488** | **0.8190** |
| XGBoost | 0.9779 | 0.9152 | 0.7276 | 0.8107 | 0.9466 | 0.8170 |
| Random Forest | 0.9775 | 0.9136 | 0.7221 | 0.8066 | 0.9360 | 0.7987 |
| Decision Tree | 0.9235 | 0.4435 | 0.6999 | 0.5430 | 0.8194 | 0.3299 |
| Logistic Regression | 0.7630 | 0.1962 | **0.8550** | 0.3191 | 0.8843 | 0.2873 |

### Semi-supervised models

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Semi Random Forest | 0.9797 | 0.9559 | 0.7211 | 0.8221 | 0.9297 | 0.7934 |
| Semi XGBoost | 0.9771 | 0.9033 | 0.7248 | 0.8043 | 0.9414 | 0.8050 |
| Semi LightGBM | 0.9781 | 0.9194 | 0.7267 | 0.8118 | 0.9373 | 0.8043 |
| Semi Soft Voting Ensemble | 0.9788 | 0.9334 | 0.7248 | 0.8160 | 0.9333 | 0.7987 |

### Supervised vs. semi-supervised F1-score

| Model | Supervised | Semi-supervised | Difference |
|---|---:|---:|---:|
| Random Forest | 0.8066 | 0.8221 | +0.0155 |
| XGBoost | 0.8107 | 0.8043 | −0.0064 |
| LightGBM | 0.8149 | 0.8118 | −0.0031 |
| Soft Voting Ensemble | 0.8173 | 0.8160 | −0.0013 |

### Unsupervised baseline

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.4643 | 0.0205 | 0.1551 | 0.0363 | 0.1897 | 0.0371 |

### Feature 104 ablation

Feature 104 was examined because it appeared among the important features. The ensemble was retrained without this feature, with the scaler and SMOTE refitted on the remaining columns.

| Ensemble | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| All features | 0.9789 | 0.9368 | 0.7248 | 0.8173 | 0.9397 | 0.8086 |
| Without feature 104 | 0.9785 | 0.9269 | 0.7258 | 0.8141 | 0.9433 | 0.8106 |

Feature 104 refers to column label 104 in the headerless features file. Column 0 is the transaction ID, and column 1 is the time step.

### Notebook visualizations

The notebook includes:

- Class distribution.
- Labeled transaction counts and illicit rate per time step.
- Degree distribution by class.
- Supervised model comparison charts.
- ROC curves and confusion matrix.
- XGBoost feature importance.
- Ensemble precision and recall per test time step.
- Feature 104 analysis.
- Pseudo-label probability distribution.
- Isolation Forest score distribution and anomaly rate.
- Supervised, semi-supervised, and unsupervised comparison.

---

##  Analysis

### Precision–recall trade-off

- **Tree-based models favor precision.** Random Forest, XGBoost, LightGBM, and the ensemble achieve precision between 0.9136 and 0.9368, with recall between 0.7221 and 0.7276.
- **Logistic Regression favors recall.** It identifies 85.50% of illicit transactions, but only 19.62% of its positive predictions are correct.
- **Accuracy alone can be misleading.** The Decision Tree achieves 0.9235 accuracy but only 0.4435 precision.
- **The threshold is fixed.** All reported classification metrics use a threshold of 0.5. Threshold tuning is not included.

### Temporal generalization

- The illicit share decreases from 11.58% in training to 6.50% in testing.
- The notebook examines ensemble precision and recall across the 15 test time steps.
- Performance differences across time may reflect distribution shift between the training and test periods.
- Removing feature 104 produces only a small change in the ensemble's results.

### Semi-supervised learning

- 157,205 transactions (77.15% of the dataset) are unlabeled.
- Pseudo-labeling produces mixed results.
- Semi-supervised Random Forest achieves the highest F1-score among the evaluated models at 0.8221.
- However, its ROC-AUC and PR-AUC are lower than those of the supervised Random Forest.
- The supervised and semi-supervised experiments use different class-balancing strategies, which limits direct comparison.

### Unsupervised anomaly detection

Isolation Forest achieves an F1-score of 0.0363 and a PR-AUC of 0.0371. Its ROC-AUC of 0.1897 is below 0.5, indicating poor ranking performance on the test set under this setup.

The test-set illicit prevalence is approximately 0.065, which provides a useful reference point for PR-AUC. Isolation Forest performs below that reference.

### Data leakage controls and considerations

| Risk | Control or consideration |
|---|---|
| Future transactions entering training | Chronological split: training ≤ time step 34, testing ≥ time step 35 |
| Test statistics affecting scaling | Scaler fitted on training data only |
| Synthetic samples entering the test set | SMOTE applied only to training data |
| Future graph information | Temporal availability rule used for the degree feature |
| Test-period unlabeled data entering pseudo-labeling | Unlabeled transactions restricted to time steps 1–34 |
| Time step acting as a shortcut | Time step excluded from model features |
| Threshold tuning on the test set | Threshold fixed at 0.5 |

**Important considerations:**

- Per-time-step analysis and feature 104 ablation use the test period for analysis. The best-performing model is identified from test results, so its ranking should be treated as exploratory rather than as an unbiased final model-selection estimate.
- There is no validation set, and results come from a single run with one random seed. Small differences between models have not been statistically tested.
- The graph-degree feature is computed from the edge list, including edges involving unlabeled transactions. The notebook does not verify whether every edge connects transactions from different time steps.

---

##  Dashboard

The project includes a Streamlit dashboard implemented in `app.py`.

**Dashboard:** [Open the live CryptoGuard Dashboard](https://cryptoguard-dashboard-h7qybj5g2jbmjpkgvdvd7v.streamlit.app/dataset)

The `dashboard_data/` folder contains CSV files used by the dashboard.

| File | Description |
|---|---|
| `class_distribution.csv` | Class labels and transaction counts |
| `dataset_summary.csv` | Dataset totals, labeled and unlabeled counts, and time-step information |

---

##  Repository Structure

```text
CryptoGuard-Dashboard/
├── dashboard_data/
│   ├── class_distribution.csv
│   └── dataset_summary.csv
├── app.py
├── CryptoGuard.pptx
├── requirements.txt
└── README.md
```

---

##  Technologies

| Area | Technologies |
|---|---|
| Programming language | Python |
| Dashboard | Streamlit |
| Data processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | scikit-learn, StandardScaler |
| Class balancing | imbalanced-learn, SMOTE |
| Supervised models | Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM |
| Semi-supervised learning | Pseudo-labeling |
| Unsupervised learning | Isolation Forest |
| Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC |
| Dataset download | kagglehub |
| Experiment environment | Jupyter Notebook, Google Colab |

---

##  Installation and Local Execution

### Run the dashboard locally

The following commands are intended for Windows:

```bash
# 1. Clone the repository
git clone https://github.com/hager-arafat-el-dabaa/CryptoGuard-Dashboard.git

# 2. Navigate to the project directory
cd CryptoGuard-Dashboard

# 3. Create a virtual environment
python -m venv .venv

# 4. Activate the virtual environment
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the dashboard
streamlit run app.py
```

### Reproduce the experiments

The experiments were performed in a Jupyter notebook on Google Colab. The notebook uses `kagglehub` to download the Elliptic dataset and requires internet access.

The main experiment dependencies include:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm kagglehub jupyter
```

Some notebook cells use Google Colab-specific download functionality and may need adjustment when run in a different environment.

---

##  Limitations

- A single chronological train/test split is used.
- There is no validation set or confidence-interval analysis.
- The classification threshold is fixed at 0.5.
- Model hyperparameters are not extensively tuned.
- Recall remains around 0.72–0.73 for the tree-based models.
- Distribution shift between training and test periods is not mitigated.
- The supervised and semi-supervised comparison is not fully controlled because SMOTE is used only in the supervised experiments.
- Isolation Forest performs poorly at separating illicit from licit transactions in this setup.
- Graph information is represented through a single degree feature.

---

##  Future Improvements

The following are proposed improvements and are not implemented in the current version:

- Add a validation period and walk-forward evaluation.
- Evaluate results across multiple random seeds.
- Analyze precision–recall curves and classification thresholds.
- Compare supervised and semi-supervised models under consistent class-balancing conditions.
- Engineer additional graph-based features.
- Investigate approaches to improve recall while maintaining useful precision.

---

##  Author

**Hager Arafat El Dabaa**

GitHub: [hager-arafat-el-dabaa](https://github.com/hager-arafat-el-dabaa)

---