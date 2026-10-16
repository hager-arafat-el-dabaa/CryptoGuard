# CryptoGuard — Illicit Bitcoin Transaction Detection

A classical Machine Learning project for detecting illicit Bitcoin transactions using the **Elliptic Bitcoin Transaction Dataset**. CryptoGuard combines supervised classification, semi-supervised learning, unsupervised anomaly detection, and temporal analysis to investigate how effectively illicit transactions can be identified in a highly imbalanced transaction dataset.

## Project Links

- **Interactive Dashboard:** [Open CryptoGuard Dashboard](ADD_LIVE_STREAMLIT_URL_HERE)
- **Project Presentation:** [View the Presentation](https://github.com/hager-arafat-el-dabaa/CryptoGuard-Dashboard/blob/main/CryptoGuard-Illicit-Bitcoin-Transaction%20%282%29.pptx)
- **Repository:** [CryptoGuard-Dashboard](https://github.com/hager-arafat-el-dabaa/CryptoGuard-Dashboard)

## Overview

Cryptocurrency transactions can be associated with illicit activities, making transaction monitoring an important application of Machine Learning.

CryptoGuard investigates the detection of illicit Bitcoin transactions using transaction features and temporal graph information. The project compares multiple classical ML approaches and examines whether incorporating carefully selected pseudo-labeled transactions improves classification performance.

The project also explores anomaly detection and evaluates how model performance changes across later time steps.

### Project Objectives

- Explore and audit the Elliptic transaction dataset.
- Analyze transaction labels, class imbalance, and temporal patterns.
- Engineer a temporal graph-degree feature from the transaction edge list.
- Build and compare supervised Machine Learning models.
- Investigate semi-supervised learning through pseudo-labeling.
- Explore unsupervised anomaly detection using Isolation Forest.
- Analyze model performance across the test period and investigate feature distribution shifts.
- Present dataset insights through an interactive dashboard.

## Dataset

CryptoGuard uses the **Elliptic Bitcoin Transaction Dataset**, obtained through Kaggle.

The dataset contains three files:

| File | Description |
|---|---|
| `elliptic_txs_features.csv` | Transaction identifiers, time steps, and transaction features |
| `elliptic_txs_classes.csv` | Transaction labels |
| `elliptic_txs_edgelist.csv` | Connections between transactions |

### Dataset Statistics

| Statistic | Value |
|---|---:|
| Total transactions | 203,769 |
| Labeled transactions | 46,564 |
| Unknown transactions | 157,205 |
| Licit transactions | 42,019 |
| Illicit transactions | 4,545 |
| Transaction edges | 234,355 |
| Time steps | 49 |
| Original feature columns | 165 |

The original feature file contains a transaction ID, a time-step column, and 165 transaction features.

### Class Distribution

| Class | Transactions | Percentage of All Transactions |
|---|---:|---:|
| Unknown | 157,205 | 77.15% |
| Licit | 42,019 | 20.62% |
| Illicit | 4,545 | 2.23% |

Only approximately **9.76% of labeled transactions are illicit**, making the classification problem highly imbalanced.

Unknown transactions do not have ground-truth labels. They are excluded from supervised training and explored separately in the semi-supervised and unsupervised experiments.

## Data Integrity and Exploration

Before modeling, the three dataset files were checked for consistency.

The integrity checks confirmed:

- No missing values in the loaded files.
- No duplicate rows in the individual files.
- Transaction IDs matched between the features and classes files.
- All transaction IDs referenced by the edge list were present in the features file.

Exploratory analysis examined class distribution, labeled transactions across time steps, illicit transaction rates, and graph-degree differences between licit and illicit transactions.

## Methodology

### 1. Temporal Graph Feature Engineering

A graph-degree feature was derived from the edge list to represent transaction connectivity.

To reduce future-information leakage, an edge was considered available only when both connected transactions existed. The feature was calculated using edges available at each transaction's own time step.

### 2. Temporal Train/Test Split

Instead of randomly splitting the data, CryptoGuard uses a chronological split to evaluate the models on later transactions.

| Dataset | Time Steps | Transactions |
|---|---|---:|
| Training | 1–34 | 29,894 |
| Testing | 35–49 | 16,670 |

The test set contains:

- 15,587 licit transactions
- 1,083 illicit transactions

There is no separate validation set in this experiment. The classification threshold is fixed at **0.5** for the reported supervised and semi-supervised comparisons.

The time-step column and transaction identifiers are excluded from the model input features.

### 3. Feature Scaling

`StandardScaler` is fitted on the training features only and then applied to the test features.

This prevents the scaler from learning statistical information from the test set.

### 4. Handling Class Imbalance

SMOTE is applied to the training set only.

Before resampling:

- Licit: 26,432
- Illicit: 3,462

After SMOTE:

- Licit: 26,432
- Illicit: 26,432

The test set retains its original class distribution.

## Machine Learning Models

### Supervised Learning

Six supervised approaches were evaluated:

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | Default configuration with a fixed random state |
| Random Forest | 200 trees |
| XGBoost | 200 estimators, learning rate 0.1, maximum depth 6 |
| LightGBM | 200 estimators, learning rate 0.1, maximum depth 6 |
| Soft Voting Ensemble | Average of Random Forest, XGBoost, and LightGBM illicit-class probabilities |

The ensemble combines the predicted probabilities from three tree-based models and classifies a transaction as illicit when the average probability reaches the 0.5 threshold.

### Semi-Supervised Learning

Pseudo-labeling was used to investigate whether unknown transactions from the training period could provide additional training information.

The procedure was:

1. Select unknown transactions from time steps 1–34.
2. Generate illicit-class probabilities using the Random Forest, XGBoost, and LightGBM models.
3. Average the probabilities to form ensemble predictions.
4. Assign pseudo-labels only to high-confidence predictions:
   - Probability ≥ 0.90: pseudo-labeled illicit.
   - Probability ≤ 0.10: pseudo-labeled licit.
5. Balance the selected pseudo-labeled transactions between the two classes.
6. Add them to the original training data.
7. Retrain the Random Forest, XGBoost, LightGBM, and Soft Voting Ensemble approaches.
8. Evaluate the resulting models on the same chronological test period.

The experiment identified:

- 2,002 high-confidence illicit predictions among unknown transactions.
- 77,320 high-confidence licit predictions.
- 27,049 uncertain predictions.
- 4,004 balanced pseudo-labeled transactions added to training.

These are model-generated labels, not verified ground-truth labels.

### Unsupervised Anomaly Detection

Isolation Forest was trained on unknown transactions from time steps 1–34, without using their class labels.

Configuration:

- `n_estimators=200`
- `contamination=0.05`

The model was then applied to the test set to investigate whether detected anomalies corresponded to illicit transactions.

An anomaly is not necessarily illicit, so Isolation Forest is treated as a complementary experiment rather than the primary classification approach.

## Evaluation Metrics

Because illicit transactions represent a minority of the labeled data, accuracy alone is insufficient.

CryptoGuard evaluates models using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted illicit transactions that are actually illicit.
- **Recall:** Proportion of actual illicit transactions detected.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank illicit transactions above licit transactions across thresholds.
- **PR-AUC:** Precision-recall performance, particularly useful for an imbalanced positive class.

The main model comparison uses the **F1-score**, while Precision, Recall, ROC-AUC, and PR-AUC provide additional insight.

## Results

All values below are from the notebook's test-set evaluation, using time steps 35–49 and a classification threshold of 0.5.

### Supervised Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Soft Voting Ensemble | 97.89% | 93.68% | 72.48% | **81.73%** | 93.97% | 80.86% |
| LightGBM | 97.85% | 92.60% | 72.76% | 81.49% | **94.88%** | **81.90%** |
| XGBoost | 97.79% | 91.52% | 72.76% | 81.07% | 94.66% | 81.70% |
| Random Forest | 97.75% | 91.36% | 72.21% | 80.66% | 93.60% | 79.87% |
| Decision Tree | 92.35% | 44.35% | 69.99% | 54.30% | 81.94% | 32.99% |
| Logistic Regression | 76.30% | 19.62% | **85.50%** | 31.91% | 88.43% | 28.73% |

**Key observations:**

- The Soft Voting Ensemble achieved the highest F1-score among the supervised models.
- LightGBM achieved the highest ROC-AUC and PR-AUC in the supervised comparison.
- Logistic Regression achieved the highest recall but had much lower precision, illustrating the trade-off between detecting more illicit transactions and generating false positives.
- The tree-based ensemble approaches provided a more balanced combination of precision and recall.

### Supervised vs. Semi-Supervised Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Semi-Supervised Random Forest | **97.97%** | **95.59%** | 72.11% | **82.21%** | 92.97% | 79.34% |
| Soft Voting Ensemble | 97.89% | 93.68% | 72.48% | 81.73% | 93.97% | 80.86% |
| Semi-Supervised Soft Voting Ensemble | 97.88% | 93.34% | 72.48% | 81.60% | 93.33% | 79.87% |
| Random Forest | 97.75% | 91.36% | 72.21% | 80.66% | 93.60% | 79.87% |

The Semi-Supervised Random Forest achieved the highest F1-score in the final comparison, improving from **80.66% to 82.21%** after incorporating pseudo-labeled training examples.

However, pseudo-labeling did not improve every model. The semi-supervised Soft Voting Ensemble achieved an F1-score of 81.60%, compared with 81.73% for its supervised counterpart.

This suggests that the effect of pseudo-labeling depends on the learning algorithm and the quality of the generated labels.

### Isolation Forest Results

| Metric | Score |
|---|---:|
| Accuracy | 46.43% |
| Precision | 2.05% |
| Recall | 15.51% |
| F1-score | 3.63% |
| ROC-AUC | 18.97% |
| PR-AUC | 3.71% |

Under this experimental setup, Isolation Forest performed poorly at identifying illicit transactions. This reinforces the distinction between detecting unusual transactions and accurately classifying transactions as illicit.

## Temporal Performance and Feature Analysis

CryptoGuard also examines how the Soft Voting Ensemble performs across the 15 test time steps.

This analysis is important because a model that performs well on aggregate can still perform inconsistently on later transaction periods.

### Feature 104 Ablation

Feature 104 appeared among the important features identified by XGBoost. Its mean value was analyzed across time steps, and the ensemble was retrained without this feature.

| Metric | With Feature 104 | Without Feature 104 |
|---|---:|---:|
| Accuracy | 97.89% | 97.85% |
| Precision | 93.68% | 92.69% |
| Recall | 72.48% | 72.58% |
| F1-score | 81.73% | 81.41% |
| ROC-AUC | 93.97% | 94.33% |
| PR-AUC | 80.86% | 81.06% |

Removing Feature 104 slightly reduced the F1-score while increasing ROC-AUC and PR-AUC. The results indicate that the feature affects different evaluation metrics differently; its removal did not produce an overall improvement across all metrics.

## Interactive Dashboard

The Streamlit dashboard presents dataset-level information and explains the models and evaluation methodology.

The dashboard is designed to communicate the project's main components, including:

- Dataset and class distribution.
- Dataset summary and temporal evaluation design.
- Supervised, semi-supervised, and unsupervised approaches.
- Model evaluation concepts and project insights.

**Dashboard:** [Open the live application](ADD_LIVE_STREAMLIT_URL_HERE)

## Technology Stack

- **Python** — Core programming language
- **Pandas & NumPy** — Data manipulation and numerical computing
- **Matplotlib & Seaborn** — Exploratory data analysis and visualization
- **Scikit-learn** — Preprocessing, classification, evaluation, and anomaly detection
- **Imbalanced-learn** — SMOTE
- **XGBoost** — Gradient boosting
- **LightGBM** — Gradient boosting
- **Streamlit** — Interactive dashboard
- **Plotly** — Dashboard visualizations

## Repository Structure

```text
CryptoGuard-Dashboard/
├── app.py
├── requirements.txt
├── dashboard_data/
│   ├── class_distribution.csv
│   └── dataset_summary.csv
└── CryptoGuard-Illicit-Bitcoin-Transaction (2).pptx
```

The full experimentation notebook is maintained separately from the dashboard repository. The dashboard uses exported CSV files for its dataset summaries.

## Limitations

- **Temporal distribution shift:** Transaction patterns can change across time steps, affecting the model's ability to generalize to later transactions.
- **Class imbalance:** Illicit transactions are a small fraction of all transactions, making false positives and false negatives important considerations.
- **Pseudo-label uncertainty:** High-confidence model predictions on unknown transactions may still be incorrect.
- **Anomaly detection limitations:** Anomalies are not necessarily illicit, and Isolation Forest performed poorly in this experiment.
- **No separate validation set:** The reported experiment uses a train/test split without an independent validation period. The results should be interpreted within this evaluation design.
- **Dataset labels:** The experiments evaluate the available Elliptic labels and do not independently establish whether real-world transactions are illicit.

## Future Improvements

- Evaluate additional temporal split strategies using a separate validation period.
- Investigate threshold selection on validation data before final testing.
- Analyze concept drift and performance degradation across time steps.
- Improve pseudo-label selection and measure its effect on generalization.
- Explore additional graph-derived features and feature selection methods.
- Improve the dashboard with model comparison charts and temporal performance visualizations.
- Evaluate operational trade-offs between recall, precision, and false-positive rates.

## Conclusion

CryptoGuard demonstrates a classical Machine Learning workflow for illicit Bitcoin transaction detection using temporal transaction data.

The experiments compare supervised classification, pseudo-labeling-based semi-supervised learning, and unsupervised anomaly detection. The Semi-Supervised Random Forest achieved the highest F1-score in the final comparison at **82.21%**, while LightGBM achieved the highest ROC-AUC and PR-AUC among the supervised models.

The temporal analysis also highlights an important practical challenge: strong aggregate metrics do not guarantee consistent performance across later transaction periods.

CryptoGuard is an experimental research and portfolio project, not a production-ready financial crime detection system.

---

**Author:** Hager Arafat El Dabaa

**Project:** CryptoGuard — Illicit Bitcoin Transaction Detection

**Focus:** Classical Machine Learning · Semi-Supervised Learning · Anomaly Detection · Temporal Analysis