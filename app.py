

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# CryptoGuard Dashboard
# Dark Olive Theme — Presentation Results
# =========================================================

st.set_page_config(
    page_title="CryptoGuard | Illicit Bitcoin Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- COLORS --------------------

OLIVE = "#3B3F1F"       # Dark Olive: all text
NAVY = "#1D3557"
TEAL = "#2A9D8F"
RED = "#E63946"
ORANGE = "#F7931A"
LIGHT = "#F4F6F8"
WHITE = "#FFFFFF"
MUTED = "#6B705C"

# -------------------- GLOBAL CSS --------------------

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        color: {OLIVE} !important;
    }}

    html, body, p, span, label, li, div,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p {{
        color: {OLIVE};
    }}

    .stApp {{
        background-color: {WHITE};
        color: {OLIVE};
    }}

    [data-testid="stSidebar"] {{
        background-color: {LIGHT};
    }}

    [data-testid="stSidebar"] * {{
        color: {OLIVE} !important;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{
        color: {OLIVE} !important;
    }}

    [data-testid="stMetric"] {{
        background-color: {LIGHT};
        border: 1px solid #E2E5D5;
        border-radius: 12px;
        padding: 14px;
    }}

    input, textarea, select {{
        color: {OLIVE} !important;
        background-color: {WHITE} !important;
    }}

    [data-baseweb="select"] *,
    [data-baseweb="input"] *,
    [data-baseweb="textarea"] * {{
        color: {OLIVE} !important;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        color: {OLIVE} !important;
    }}

    button, [data-testid="stFileUploader"] {{
        color: {OLIVE};
    }}

    [data-testid="stFileUploader"] section {{
        background-color: {LIGHT};
        border: 1px dashed {OLIVE};
    }}

    hr {{
        border-color: #D8DBC9;
    }}

    .cg-card {{
        background-color: {LIGHT};
        border: 1px solid #E2E5D5;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }}

    .cg-subtitle {{
        color: {OLIVE};
        font-size: 15px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- PRESENTATION RESULTS --------------------
# Values below are copied from the project presentation.

results = pd.DataFrame({
    "Model": [
        "Semi-Supervised RF",
        "Soft Voting Ensemble",
        "LightGBM",
        "XGBoost",
        "Random Forest",
        "Decision Tree",
        "Logistic Regression",
        "Isolation Forest"
    ],
    "Accuracy": [97.97, 97.89, 97.85, 97.79, 97.75, 92.35, 76.30, 46.43],
    "Precision": [95.59, 93.68, 92.60, 91.52, 91.36, 44.35, 19.62, 2.05],
    "Recall": [72.11, 72.48, 72.76, 72.76, 72.21, 69.99, 85.50, 15.51],
    "F1-Score": [82.21, 81.73, 81.49, 81.07, 80.66, 54.30, 31.91, 3.63],
    "ROC-AUC": [92.97, 93.97, 94.88, 94.66, 93.60, 81.94, 88.43, None]
})

TOTAL_TRANSACTIONS = 203769
TOTAL_EDGES = 234355
LABELED = 46564
UNKNOWN = 157205
LICIT = 42019
ILLICIT = 4545
TIME_STEPS = 49
TEST_LABELED = 16670
TEST_ILLICIT = 1083


# -------------------- CHART STYLE FUNCTION --------------------

def style_chart(fig, height=430):
    """
    Apply Dark Olive to all Plotly chart text:
    titles, axes, tick labels, legends, annotations,
    hover labels, and color bars.
    """

    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color=OLIVE
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=19,
            color=OLIVE
        ),
        legend=dict(
            font=dict(color=OLIVE, size=12),
            title_font=dict(color=OLIVE)
        ),
        hoverlabel=dict(
            bgcolor=WHITE,
            bordercolor=OLIVE,
            font=dict(
                family="Arial, sans-serif",
                color=OLIVE,
                size=13
            )
        ),
        margin=dict(l=35, r=25, t=70, b=45)
    )

    fig.update_xaxes(
        title_font=dict(color=OLIVE, size=13),
        tickfont=dict(color=OLIVE, size=11),
        showline=True,
        linecolor="#D8DBC9",
        gridcolor="#E8EADD",
        zerolinecolor="#D8DBC9"
    )

    fig.update_yaxes(
        title_font=dict(color=OLIVE, size=13),
        tickfont=dict(color=OLIVE, size=11),
        showline=True,
        linecolor="#D8DBC9",
        gridcolor="#E8EADD",
        zerolinecolor="#D8DBC9"
    )

    # Text annotations inside charts
    fig.update_annotations(
        font=dict(color=OLIVE, size=12)
    )

    # Colorbar text, if the chart uses a color scale
    for trace in fig.data:
        if hasattr(trace, "colorbar") and trace.colorbar:
            trace.colorbar.tickfont = dict(color=OLIVE)
            trace.colorbar.title.font = dict(color=OLIVE)

    return fig


def show_chart(fig, height=430):
    """Style every chart before displaying it."""
    fig = style_chart(fig, height)
    st.plotly_chart(fig, use_container_width=True)


def show_olive_table(dataframe):
    """Style dataframe cell text with Dark Olive."""
    styled = dataframe.style.set_properties(
        **{
            "color": OLIVE,
            "background-color": WHITE,
            "border-color": "#E2E5D5"
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f'<p class="cg-subtitle">{subtitle}</p>',
            unsafe_allow_html=True
        )


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("# 🛡️ CryptoGuard")
    st.markdown("### Illicit Bitcoin Transaction Detection")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Model Comparison",
            "Dataset Explorer",
            "Temporal Analysis",
            "Risk Review"
        ]
    )

    st.divider()
    st.caption("Dark Olive Theme")
    st.caption("Project results follow the final presentation.")


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "Overview":

    st.title("🛡️ CryptoGuard Dashboard")
    st.markdown(
        "### Illicit Bitcoin Transaction Detection"
    )
    st.markdown(
        "Explore dataset statistics, model performance, "
        "temporal evaluation, and transaction risk."
    )

    st.divider()

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", f"{TOTAL_TRANSACTIONS:,}")
    c2.metric("Graph Edges", f"{TOTAL_EDGES:,}")
    c3.metric("Labeled Transactions", f"{LABELED:,}")
    c4.metric("Time Steps", f"{TIME_STEPS}")

    st.markdown(" ")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Licit", f"{LICIT:,}")
    c6.metric("Illicit", f"{ILLICIT:,}")
    c7.metric("Unknown Labels", f"{UNKNOWN:,}")
    c8.metric("Test Illicit", f"{TEST_ILLICIT:,}")

    st.divider()

    left, right = st.columns(2)

    # Model F1 comparison
    with left:
        section_title("Model Performance", "F1-score comparison")

        f1_data = results.sort_values("F1-Score", ascending=True)

        fig = px.bar(
            f1_data,
            x="F1-Score",
            y="Model",
            orientation="h",
            text="F1-Score",
            color="F1-Score",
            color_continuous_scale=[
                "#E8EADD",
                TEAL,
                NAVY
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            textfont=dict(color=OLIVE),
            cliponaxis=False
        )

        fig.update_layout(
            xaxis_title="F1-Score (%)",
            yaxis_title="Model",
            coloraxis_showscale=False
        )

        show_chart(fig, 450)

    # Dataset label distribution
    with right:
        section_title("Labeled Data Distribution")

        label_data = pd.DataFrame({
            "Class": ["Licit", "Illicit"],
            "Transactions": [LICIT, ILLICIT]
        })

        fig = px.pie(
            label_data,
            names="Class",
            values="Transactions",
            hole=0.55,
            color="Class",
            color_discrete_map={
                "Licit": TEAL,
                "Illicit": RED
            }
        )

        fig.update_traces(
            textinfo="label+percent",
            textfont=dict(color=OLIVE, size=13),
            marker=dict(line=dict(color=WHITE, width=2))
        )

        fig.update_layout(
            legend=dict(
                font=dict(color=OLIVE),
                title_font=dict(color=OLIVE)
            )
        )

        show_chart(fig, 450)

    st.divider()

    section_title("Best Model by F1-Score")

    best_model = results.loc[results["F1-Score"].idxmax()]

    st.markdown(
        f"""
        <div class="cg-card">
            <h3 style="color:{OLIVE};">
                {best_model["Model"]}
            </h3>
            <p style="color:{OLIVE};">
                F1-Score: <b>{best_model["F1-Score"]:.2f}%</b>
            </p>
            <p style="color:{OLIVE};">
                Precision: <b>{best_model["Precision"]:.2f}%</b>
                &nbsp; | &nbsp;
                Recall: <b>{best_model["Recall"]:.2f}%</b>
            </p>
            <p style="color:{OLIVE};">
                Accuracy: <b>{best_model["Accuracy"]:.2f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2 — MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")
    st.markdown(
        "Compare the models using the final presentation metrics."
    )

    metric = st.selectbox(
        "Choose a metric",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    )

    plot_data = results.dropna(subset=[metric]).sort_values(
        metric,
        ascending=True
    )

    fig = px.bar(
        plot_data,
        x=metric,
        y="Model",
        orientation="h",
        text=metric,
        color=metric,
        color_continuous_scale=[
            "#E8EADD",
            TEAL,
            NAVY
        ]
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont=dict(color=OLIVE),
        cliponaxis=False
    )

    fig.update_layout(
        title=f"{metric} Across Models",
        xaxis_title=f"{metric} (%)",
        yaxis_title="Model",
        coloraxis_showscale=False
    )

    show_chart(fig, 520)

    st.divider()

    section_title("Compare Multiple Metrics")

    metric_options = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]

    selected_metrics = st.multiselect(
        "Select metrics to compare",
        metric_options,
        default=["Precision", "Recall", "F1-Score"]
    )

    if selected_metrics:
        comparison = results.melt(
            id_vars="Model",
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Score"
        ).dropna(subset=["Score"])

        fig = px.bar(
            comparison,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=[
                NAVY,
                TEAL,
                ORANGE,
                RED,
                MUTED
            ],
            hover_data={"Score": ":.2f"}
        )

        fig.update_layout(
            title="Selected Metrics by Model",
            xaxis_title="Model",
            yaxis_title="Score (%)",
            xaxis_tickangle=-30,
            legend_title_text="Metric"
        )

        show_chart(fig, 500)

    st.divider()

    section_title("Full Results Table")
    show_olive_table(results.round(2))


# =========================================================
# PAGE 3 — DATASET EXPLORER
# =========================================================

elif page == "Dataset Explorer":

    st.title("🗂️ Dataset Explorer")
    st.markdown(
        "Explore the dataset summary or upload your own CSV file."
    )

    st.info(
        "The summary values below describe the project dataset. "
        "Upload a CSV to explore its actual columns and records."
    )

    summary = pd.DataFrame({
        "Category": [
            "Total Transactions",
            "Graph Edges",
            "Labeled Transactions",
            "Unknown Labels",
            "Licit Transactions",
            "Illicit Transactions",
            "Time Steps",
            "Test Labeled Transactions",
            "Test Illicit Transactions"
        ],
        "Count": [
            TOTAL_TRANSACTIONS,
            TOTAL_EDGES,
            LABELED,
            UNKNOWN,
            LICIT,
            ILLICIT,
            TIME_STEPS,
            TEST_LABELED,
            TEST_ILLICIT
        ]
    })

    left, right = st.columns(2)

    with left:
        section_title("Dataset Summary")
        show_olive_table(summary)

    with right:
        section_title("Label Counts")

        label_df = pd.DataFrame({
            "Label": ["Licit", "Illicit", "Unknown"],
            "Count": [LICIT, ILLICIT, UNKNOWN]
        })

        fig = px.bar(
            label_df,
            x="Label",
            y="Count",
            text="Count",
            color="Label",
            color_discrete_map={
                "Licit": TEAL,
                "Illicit": RED,
                "Unknown": ORANGE
            }
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            textfont=dict(color=OLIVE),
            cliponaxis=False
        )

        fig.update_layout(
            title="Transaction Label Distribution",
            xaxis_title="Class",
            yaxis_title="Number of Transactions",
            showlegend=False
        )

        show_chart(fig, 400)

    st.divider()

    section_title("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success("CSV loaded successfully.")

            a, b, c = st.columns(3)
            a.metric("Rows", f"{df.shape[0]:,}")
            b.metric("Columns", f"{df.shape[1]:,}")
            c.metric(
                "Missing Cells",
                f"{int(df.isna().sum().sum()):,}"
            )

            st.markdown("### Preview")
            show_olive_table(df.head(20))

            st.markdown("### Column Information")

            column_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [
                    str(df[col].dtype) for col in df.columns
                ],
                "Missing Values": [
                    int(df[col].isna().sum()) for col in df.columns
                ],
                "Unique Values": [
                    int(df[col].nunique(dropna=True))
                    for col in df.columns
                ]
            })

            show_olive_table(column_info)

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                st.divider()
                section_title("Explore a Numeric Column")

                chosen_column = st.selectbox(
                    "Choose a column",
                    numeric_columns
                )

                values = df[chosen_column].dropna()

                if not values.empty:
                    fig = px.histogram(
                        x=values,
                        nbins=40,
                        labels={
                            "x": chosen_column,
                            "count": "Frequency"
                        },
                        color_discrete_sequence=[TEAL]
                    )

                    fig.update_layout(
                        title=f"Distribution of {chosen_column}",
                        xaxis_title=chosen_column,
                        yaxis_title="Frequency"
                    )

                    show_chart(fig, 400)

                    st.markdown("### Descriptive Statistics")

                    stats = (
                        values.describe()
                        .rename_axis("Statistic")
                        .reset_index(name="Value")
                    )

                    show_olive_table(stats.round(4))

            categorical_columns = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            if categorical_columns:
                st.divider()
                section_title("Explore a Categorical Column")

                chosen_category = st.selectbox(
                    "Choose a categorical column",
                    categorical_columns
                )

                counts = (
                    df[chosen_category]
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .rename_axis("Category")
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    counts,
                    x="Category",
                    y="Count",
                    text="Count",
                    color_discrete_sequence=[TEAL]
                )

                fig.update_traces(
                    textposition="outside",
                    textfont=dict(color=OLIVE),
                    cliponaxis=False
                )

                fig.update_layout(
                    title=f"Top Categories: {chosen_category}",
                    xaxis_title=chosen_category,
                    yaxis_title="Count",
                    xaxis_tickangle=-30
                )

                show_chart(fig, 430)

        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

    else:
        st.caption(
            "No CSV uploaded yet. The project summary remains available above."
        )


# =========================================================
# PAGE 4 — TEMPORAL ANALYSIS
# =========================================================

elif page == "Temporal Analysis":

    st.title("⏳ Temporal Analysis")

    st.markdown(
        "Review the time-based evaluation setup and the "
        "reported feature-removal experiment."
    )

    st.info(
        "This page shows the evaluation design and reported results. "
        "It does not invent per-time-step predictions or transaction counts."
    )

    section_title("Temporal Evaluation")

    # Time ranges based on the project notebook's split setup.
    periods = pd.DataFrame({
        "Phase": ["Training", "Validation", "Testing"],
        "Start": [1, 35, 42],
        "End": [34, 41, 49],
        "Color": [NAVY, ORANGE, TEAL]
    })

    fig = go.Figure()

    for _, row in periods.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["Phase"],
                x=[row["End"] - row["Start"] + 1],
                y=[row["Phase"]],
                base=[row["Start"]],
                orientation="h",
                marker_color=row["Color"],
                text=[
                    f'Time steps {row["Start"]}–{row["End"]}'
                ],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=WHITE),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Start: %{base}<br>"
                    "Duration: %{x} time steps"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        barmode="overlay",
        title="Temporal Train / Validation / Test Split",
        xaxis_title="Time Step",
        yaxis_title="Evaluation Phase",
        showlegend=False,
        xaxis=dict(range=[1, 50], dtick=5)
    )

    show_chart(fig, 330)

    st.divider()

    section_title("Feature 104 Removal Experiment")

    # Reported values from the project presentation/code summary.
    experiment = pd.DataFrame({
        "Experiment": [
            "Without Feature 104"
        ],
        "F1-Score": [81.41],
        "ROC-AUC": [94.33]
    })

    c1, c2 = st.columns(2)

    c1.metric(
        "Reported F1-Score",
        f'{experiment.loc[0, "F1-Score"]:.2f}%'
    )

    c2.metric(
        "Reported ROC-AUC",
        f'{experiment.loc[0, "ROC-AUC"]:.2f}%'
    )

    st.markdown(
        """
        Feature 104 is an anonymized numerical feature.
        Its real-world meaning is not specified by the dataset, so
        analysis should focus on its statistical behavior and effect
        on model performance rather than assigning it an unsupported meaning.
        """
    )

    st.divider()

    section_title("Temporal Evaluation Notes")

    notes = pd.DataFrame({
        "Item": [
            "Training period",
            "Validation period",
            "Testing period",
            "Total time steps"
        ],
        "Value": [
            "1–34",
            "35–41",
            "42–49",
            str(TIME_STEPS)
        ]
    })

    show_olive_table(notes)


# =========================================================
# PAGE 5 — RISK REVIEW
# =========================================================

elif page == "Risk Review":

    st.title("🚨 Risk Review")

    st.markdown(
        "Upload a CSV containing transaction records and a model score "
        "or predicted probability to review higher-risk transactions."
    )

    st.warning(
        "This page ranks the scores in your uploaded file. "
        "It does not run the trained CryptoGuard model by itself."
    )

    uploaded_risk_file = st.file_uploader(
        "Upload scored transactions",
        type=["csv"],
        key="risk_csv"
    )

    if uploaded_risk_file is not None:
        try:
            risk_df = pd.read_csv(uploaded_risk_file)

            st.markdown("### Uploaded Data")
            st.caption(
                f'{len(risk_df):,} records · {len(risk_df.columns):,} columns'
            )

            numeric_columns = risk_df.select_dtypes(
                include="number"
            ).columns.tolist()

            if not numeric_columns:
                st.error(
                    "No numeric columns found. Upload a CSV with a numeric "
                    "risk score or predicted probability."
                )

            else:
                score_column = st.selectbox(
                    "Select the model score / probability column",
                    numeric_columns
                )

                threshold = st.slider(
                    "Risk threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help=(
                        "Use this setting for probability columns "
                        "scaled between 0 and 1."
                    )
                )

                score_values = pd.to_numeric(
                    risk_df[score_column],
                    errors="coerce"
                )

                valid_scores = score_values.dropna()

                if valid_scores.empty:
                    st.error(
                        "The selected column has no valid numeric scores."
                    )

                else:
                    # For probabilities, use the threshold directly.
                    # For percentage scores, convert the threshold to %.
                    if valid_scores.min() >= 0 and valid_scores.max() <= 1:
                        applied_threshold = threshold
                    elif valid_scores.min() >= 0 and valid_scores.max() <= 100:
                        applied_threshold = threshold * 100
                    else:
                        applied_threshold = threshold

                    review_df = risk_df.copy()
                    review_df["_RiskScore"] = score_values

                    review_df["Risk Category"] = "Lower score"
                    review_df.loc[
                        review_df["_RiskScore"] >= applied_threshold,
                        "Risk Category"
                    ] = "Higher score"

                    higher = review_df[
                        review_df["Risk Category"] == "Higher score"
                    ]

                    low = review_df[
                        review_df["Risk Category"] == "Lower score"
                    ]

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Records Reviewed",
                        f"{len(review_df):,}"
                    )

                    m2.metric(
                        "Higher-Score Records",
                        f"{len(higher):,}"
                    )

                    m3.metric(
                        "Lower-Score Records",
                        f"{len(low):,}"
                    )

                    st.divider()

                    left, right = st.columns(2)

                    with left:
                        section_title("Risk Score Distribution")

                        fig = px.histogram(
                            review_df,
                            x="_RiskScore",
                            nbins=40,
                            color_discrete_sequence=[TEAL]
                        )

                        fig.add_vline(
                            x=applied_threshold,
                            line_dash="dash",
                            line_color=RED,
                            annotation_text="Threshold",
                            annotation_font_color=OLIVE
                        )

                        fig.update_layout(
                            title="Distribution of Selected Scores",
                            xaxis_title=score_column,
                            yaxis_title="Number of Records"
                        )

                        show_chart(fig, 400)

                    with right:
                        section_title("Score Categories")

                        category_counts = (
                            review_df["Risk Category"]
                            .value_counts()
                            .reindex(
                                ["Higher score", "Lower score"],
                                fill_value=0
                            )
                            .rename_axis("Category")
                            .reset_index(name="Count")
                        )

                        fig = px.pie(
                            category_counts,
                            names="Category",
                            values="Count",
                            hole=0.5,
                            color="Category",
                            color_discrete_map={
                                "Higher score": RED,
                                "Lower score": TEAL
                            }
                        )

                        fig.update_traces(
                            textinfo="label+percent",
                            textfont=dict(color=OLIVE)
                        )

                        show_chart(fig, 400)

                    st.divider()

                    section_title("Higher-Score Records")

                    sorted_higher = higher.sort_values(
                        "_RiskScore",
                        ascending=False
                    )

                    show_olive_table(sorted_higher.head(100))

                    csv_data = sorted_higher.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Higher-Score Records",
                        data=csv_data,
                        file_name="cryptoguard_risk_review.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Could not process the uploaded file: {error}")

    else:
        st.info(
            "Upload a scored-transactions CSV to start reviewing records."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    f"""
    <div style="text-align:center; color:{OLIVE}; padding:10px;">
        <b>CryptoGuard</b> — Illicit Bitcoin Transaction Detection
        <br>
        <span style="color:{OLIVE};">
            Dark Olive Edition · Results based on the project presentation
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
