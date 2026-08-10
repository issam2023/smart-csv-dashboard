import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from scipy import stats


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Data Analytics | A. Masmi",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 1.15rem;
    opacity: .75;
}

.developer {
    margin-top: 15px;
    font-size: .95rem;
    opacity: .8;
}

.workflow {
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
    margin-top: 10px;
    margin-bottom: 22px;
    text-align: center;
    font-size: 1.05rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DYNAMIC SESSION INFORMATION
# ============================================================

now = datetime.now()

session_date = now.strftime("%B %d, %Y")
session_time = now.strftime("%I:%M %p")


# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="hero">

<div class="hero-title">
📊 Smart Data Analytics & Preprocessing
</div>

<div class="hero-subtitle">
Professional CSV Cleaning, Distribution Analysis,
Visualization and Data Preparation Dashboard
</div>

<div class="developer">
<b>Developed by A. Masmi</b><br>
jovina@gmx.us<br>
Session: {session_date} • {session_time}
</div>

</div>
""", unsafe_allow_html=True)


st.write(
    """
    Upload a CSV dataset to inspect its quality, study numeric
    distributions, detect potential problems, clean and preprocess
    the data, explore relationships, and export a prepared dataset.
    """
)

st.markdown("""
<div class="workflow">
CSV Upload → Diagnose → Distribution Analysis → Clean → Visualize → Export
</div>
""", unsafe_allow_html=True)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload a CSV file to start the analysis."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df_original = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read this CSV file: {e}"
    )

    st.stop()


df = df_original.copy()


# ============================================================
# BASIC INFORMATION
# ============================================================

rows = len(df)
columns = len(df.columns)

missing = int(
    df.isna().sum().sum()
)

duplicates = int(
    df.duplicated().sum()
)

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(data):

    if data.empty:
        return 0

    score = 100

    total_cells = max(
        data.shape[0] * data.shape[1],
        1
    )

    missing_ratio = (
        data.isna().sum().sum()
        / total_cells
    )

    duplicate_ratio = (
        data.duplicated().sum()
        / max(len(data), 1)
    )

    score -= missing_ratio * 50
    score -= duplicate_ratio * 30

    empty_columns = (
        data.isna().all().sum()
    )

    constant_columns = sum(
        data[col].nunique(
            dropna=True
        ) <= 1
        for col in data.columns
    )

    if len(data.columns) > 0:

        score -= (
            empty_columns
            / len(data.columns)
        ) * 10

        score -= (
            constant_columns
            / len(data.columns)
        ) * 10

    return max(
        0,
        min(100, round(score))
    )


health_score = calculate_health_score(df)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Rows",
    f"{rows:,}"
)

c2.metric(
    "Columns",
    columns
)

c3.metric(
    "Missing Values",
    f"{missing:,}"
)

c4.metric(
    "Duplicate Rows",
    f"{duplicates:,}"
)

c5.metric(
    "Data Health",
    f"{health_score}/100"
)


st.divider()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏠 Overview",
    "🩺 Data Health",
    "📈 Distribution Analysis",
    "🧹 Clean & Preprocess",
    "📊 Visual Analytics",
    "🔥 Correlations",
    "🔎 Filter",
    "💾 Export"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Dataset Overview")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Column Profile")

    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Missing %":
            (df.isna().mean() * 100)
            .round(2)
            .values,
        "Unique":
            df.nunique(
                dropna=True
            ).values
    })

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Descriptive Statistics"
    )

    try:

        description = (
            df.describe(
                include="all"
            ).T
        )

        st.dataframe(
            description,
            use_container_width=True
        )

    except Exception:

        st.warning(
            "Summary statistics could not be generated."
        )


# ============================================================
# DATA HEALTH
# ============================================================

with tabs[1]:

    st.header("Data Health Assessment")

    st.metric(
        "Overall Data Health Score",
        f"{health_score}/100"
    )

    if health_score >= 90:

        st.success(
            "Excellent overall data quality."
        )

    elif health_score >= 75:

        st.info(
            "Good data quality with some preprocessing recommended."
        )

    elif health_score >= 50:

        st.warning(
            "Several quality problems should be reviewed."
        )

    else:

        st.error(
            "Significant preprocessing is recommended."
        )


    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing":
            df.isna().sum().values,
        "Missing %":
            (
                df.isna().mean()
                * 100
            ).round(2).values,
        "Unique":
            df.nunique(
                dropna=True
            ).values,
        "Type":
            df.dtypes.astype(str).values
    })

    st.subheader(
        "Column Quality Report"
    )

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )


    missing_quality = quality[
        quality["Missing"] > 0
    ]

    if not missing_quality.empty:

        fig_missing = px.bar(
            missing_quality,
            x="Column",
            y="Missing %",
            title="Missing Data by Column"
        )

        st.plotly_chart(
            fig_missing,
            use_container_width=True
        )

    else:

        st.success(
            "No missing values detected."
        )


    st.subheader(
        "Potential Issues"
    )

    empty_cols = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    constant_cols = [
        col for col in df.columns
        if df[col].nunique(
            dropna=True
        ) <= 1
    ]

    st.write(
        f"**Duplicate rows:** {duplicates:,}"
    )

    st.write(
        f"**Completely empty columns:** {len(empty_cols)}"
    )

    st.write(
        f"**Constant columns:** {len(constant_cols)}"
    )

    if constant_cols:

        st.write(
            "Columns:",
            constant_cols
        )


# ============================================================
# DISTRIBUTION ANALYSIS
# ============================================================

with tabs[2]:

    st.header(
        "Numeric Distribution Analysis"
    )

    st.write(
        """
        Examine the shape of numeric variables before deciding
        how to preprocess them.
        """
    )

    if not numeric_cols:

        st.warning(
            "No numeric columns were detected."
        )

    else:

        selected = st.selectbox(
            "Select numeric feature",
            numeric_cols,
            key="distribution_feature"
        )

        values = (
            df[selected]
            .dropna()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(values) < 3:

            st.warning(
                "Not enough valid observations for distribution analysis."
            )

        else:

            mean = values.mean()
            median = values.median()
            std = values.std()
            skew = values.skew()
            kurtosis = values.kurtosis()

            q1 = values.quantile(.25)
            q3 = values.quantile(.75)
            iqr = q3 - q1

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "Mean",
                f"{mean:,.3f}"
            )

            d2.metric(
                "Median",
                f"{median:,.3f}"
            )

            d3.metric(
                "Std. Deviation",
                f"{std:,.3f}"
            )

            d4.metric(
                "Skewness",
                f"{skew:,.3f}"
            )


            if abs(skew) < .5:

                distribution_label = (
                    "Approximately symmetric"
                )

                st.success(
                    "Distribution appears approximately symmetric."
                )

            elif skew >= .5:

                distribution_label = (
                    "Right-skewed"
                )

                st.warning(
                    "Distribution is right-skewed."
                )

            else:

                distribution_label = (
                    "Left-skewed"
                )

                st.warning(
                    "Distribution is left-skewed."
                )


            st.write(
                f"**Distribution interpretation:** "
                f"{distribution_label}"
            )

            st.write(
                f"**Kurtosis:** {kurtosis:.3f}"
            )

            st.write(
                f"**Q1:** {q1:,.3f}"
            )

            st.write(
                f"**Q3:** {q3:,.3f}"
            )

            st.write(
                f"**IQR:** {iqr:,.3f}"
            )


            fig_hist = px.histogram(
                df,
                x=selected,
                marginal="box",
                nbins=40,
                title=(
                    f"Distribution of {selected}"
                )
            )

            fig_hist.add_vline(
                x=mean,
                line_dash="dash",
                annotation_text="Mean"
            )

            fig_hist.add_vline(
                x=median,
                line_dash="dot",
                annotation_text="Median"
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )


            st.subheader(
                "Normal Q-Q Plot"
            )

            qq_sample = values

            if len(qq_sample) > 5000:

                qq_sample = qq_sample.sample(
                    5000,
                    random_state=42
                )

            theoretical, ordered = stats.probplot(
                qq_sample,
                dist="norm",
                fit=False
            )

            qq_df = pd.DataFrame({
                "Theoretical Quantiles":
                    theoretical,
                "Observed Values":
                    ordered
            })

            fig_qq = px.scatter(
                qq_df,
                x="Theoretical Quantiles",
                y="Observed Values",
                title=f"Q-Q Plot: {selected}"
            )

            slope, intercept, _ = stats.probplot(
                qq_sample,
                dist="norm",
                fit=True
            )[1]

            x_line = np.array([
                min(theoretical),
                max(theoretical)
            ])

            y_line = (
                slope * x_line
                + intercept
            )

            fig_qq.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Normal Reference"
                )
            )

            st.plotly_chart(
                fig_qq,
                use_container_width=True
            )


            st.subheader(
                "Normality Assessment"
            )

            test_sample = values

            if len(test_sample) > 5000:

                test_sample = (
                    test_sample.sample(
                        5000,
                        random_state=42
                    )
                )

            if len(test_sample) >= 8:

                stat, p_value = (
                    stats.normaltest(
                        test_sample
                    )
                )

                n1, n2 = st.columns(2)

                n1.metric(
                    "Normality Test Statistic",
                    f"{stat:.3f}"
                )

                n2.metric(
                    "p-value",
                    f"{p_value:.5f}"
                )

                if p_value >= .05:

                    st.info(
                        """
                        The test does not provide strong evidence
                        against normality at the 5% significance level.
                        """
                    )

                else:

                    st.warning(
                        """
                        The test indicates evidence that the data
                        differ from a normal distribution.
                        """
                    )

                st.caption(
                    """
                    Statistical normality tests can become very sensitive
                    with large datasets. Interpret them together with the
                    histogram, Q-Q plot, skewness and domain context.
                    """
                )


            st.subheader(
                "Preprocessing Recommendation"
            )

            if abs(skew) > 1:

                st.info(
                    """
                    Strong skew detected. Median imputation is usually
                    more robust than mean imputation when missing values
                    and outliers are present.
                    """
                )

            else:

                st.info(
                    """
                    The feature is not strongly skewed. Mean or median
                    imputation may both be reasonable depending on the
                    analytical context.
                    """
                )


# ============================================================
# CLEAN & PREPROCESS
# ============================================================

with tabs[3]:

    st.header(
        "Data Cleaning & Preprocessing Service"
    )

    st.write(
        """
        Configure preprocessing operations below.
        The original uploaded dataset remains unchanged.
        """
    )


    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True
    )


    drop_empty = st.checkbox(
        "Drop completely empty columns",
        value=True
    )


    drop_constant = st.checkbox(
        "Drop constant columns",
        value=False
    )


    st.subheader(
        "Missing Values"
    )

    numeric_strategy = st.selectbox(
        "Numeric missing values",
        [
            "Median",
            "Mean",
            "Drop rows",
            "Keep missing"
        ]
    )

    categorical_strategy = st.selectbox(
        "Categorical missing values",
        [
            "Most frequent",
            "Unknown",
            "Drop rows",
            "Keep missing"
        ]
    )


    st.subheader(
        "Outlier Handling"
    )

    detect_outliers = st.checkbox(
        "Detect numeric outliers using IQR",
        value=True
    )

    remove_outliers = st.checkbox(
        "Remove detected IQR outliers",
        value=False
    )


    if st.button(
        "🚀 Run Data Cleaning",
        type="primary",
        use_container_width=True
    ):

        cleaned = df_original.copy()

        report = {
            "Original rows":
                len(cleaned),

            "Original columns":
                len(cleaned.columns),

            "Original missing values":
                int(
                    cleaned.isna()
                    .sum()
                    .sum()
                )
        }


        before = len(cleaned)

        if remove_duplicates:

            cleaned = (
                cleaned
                .drop_duplicates()
            )

        duplicates_removed = (
            before - len(cleaned)
        )


        empty_removed = 0

        if drop_empty:

            empty_columns = [
                col
                for col in cleaned.columns
                if cleaned[col]
                .isna()
                .all()
            ]

            empty_removed = len(
                empty_columns
            )

            cleaned = cleaned.drop(
                columns=empty_columns
            )


        constant_removed = 0

        if drop_constant:

            constant_columns = [
                col
                for col in cleaned.columns
                if cleaned[col]
                .nunique(
                    dropna=True
                ) <= 1
            ]

            constant_removed = len(
                constant_columns
            )

            cleaned = cleaned.drop(
                columns=constant_columns
            )


        clean_numeric = (
            cleaned
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        if numeric_strategy == "Median":

            for col in clean_numeric:

                if cleaned[col].isna().any():

                    cleaned[col] = (
                        cleaned[col]
                        .fillna(
                            cleaned[col]
                            .median()
                        )
                    )

        elif numeric_strategy == "Mean":

            for col in clean_numeric:

                if cleaned[col].isna().any():

                    cleaned[col] = (
                        cleaned[col]
                        .fillna(
                            cleaned[col]
                            .mean()
                        )
                    )

        elif numeric_strategy == "Drop rows":

            cleaned = (
                cleaned
                .dropna(
                    subset=clean_numeric
                )
            )


        clean_categorical = (
            cleaned
            .select_dtypes(
                exclude=np.number
            )
            .columns
        )

        if categorical_strategy == "Most frequent":

            for col in clean_categorical:

                if cleaned[col].isna().any():

                    mode = (
                        cleaned[col]
                        .mode(
                            dropna=True
                        )
                    )

                    if not mode.empty:

                        cleaned[col] = (
                            cleaned[col]
                            .fillna(
                                mode.iloc[0]
                            )
                        )

        elif categorical_strategy == "Unknown":

            for col in clean_categorical:

                cleaned[col] = (
                    cleaned[col]
                    .fillna("Unknown")
                )

        elif categorical_strategy == "Drop rows":

            cleaned = (
                cleaned
                .dropna(
                    subset=clean_categorical
                )
            )


        outliers_detected = 0
        rows_before_outliers = len(cleaned)

        if detect_outliers:

            numeric_clean_cols = (
                cleaned
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            outlier_mask = pd.Series(
                False,
                index=cleaned.index
            )

            for col in numeric_clean_cols:

                q1 = cleaned[col].quantile(.25)
                q3 = cleaned[col].quantile(.75)

                iqr = q3 - q1

                if (
                    pd.isna(iqr)
                    or iqr == 0
                ):
                    continue

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                col_outliers = (
                    (cleaned[col] < lower)
                    |
                    (cleaned[col] > upper)
                )

                outlier_mask = (
                    outlier_mask
                    | col_outliers
                )

            outliers_detected = int(
                outlier_mask.sum()
            )

            if remove_outliers:

                cleaned = cleaned.loc[
                    ~outlier_mask
                ].copy()


        outliers_removed = (
            rows_before_outliers
            - len(cleaned)
        )


        report.update({
            "Cleaned rows":
                len(cleaned),

            "Cleaned columns":
                len(cleaned.columns),

            "Duplicates removed":
                duplicates_removed,

            "Empty columns removed":
                empty_removed,

            "Constant columns removed":
                constant_removed,

            "Outliers detected":
                outliers_detected,

            "Outliers removed":
                outliers_removed,

            "Remaining missing values":
                int(
                    cleaned
                    .isna()
                    .sum()
                    .sum()
                )
        })


        st.session_state[
            "cleaned_df"
        ] = cleaned

        st.session_state[
            "cleaning_report"
        ] = report


    if "cleaned_df" in st.session_state:

        cleaned = (
            st.session_state[
                "cleaned_df"
            ]
        )

        report = (
            st.session_state[
                "cleaning_report"
            ]
        )

        st.success(
            "Data cleaning completed successfully."
        )


        before_health = (
            calculate_health_score(
                df_original
            )
        )

        after_health = (
            calculate_health_score(
                cleaned
            )
        )


        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Original Rows",
            f"{len(df_original):,}"
        )

        r2.metric(
            "Cleaned Rows",
            f"{len(cleaned):,}"
        )

        r3.metric(
            "Health Before",
            f"{before_health}/100"
        )

        r4.metric(
            "Health After",
            f"{after_health}/100"
        )


        # ====================================================
        # BEFORE VS AFTER
        # ====================================================

        st.subheader(
            "Before vs After Cleaning"
        )

        before_missing = int(
            df_original
            .isna()
            .sum()
            .sum()
        )

        after_missing = int(
            cleaned
            .isna()
            .sum()
            .sum()
        )

        before_duplicates = int(
            df_original
            .duplicated()
            .sum()
        )

        after_duplicates = int(
            cleaned
            .duplicated()
            .sum()
        )

        comparison_df = pd.DataFrame({
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
                "Data Health Score"
            ],

            "Before": [
                len(df_original),
                len(df_original.columns),
                before_missing,
                before_duplicates,
                before_health
            ],

            "After": [
                len(cleaned),
                len(cleaned.columns),
                after_missing,
                after_duplicates,
                after_health
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        compare_long = (
            comparison_df
            .melt(
                id_vars="Metric",
                value_vars=[
                    "Before",
                    "After"
                ],
                var_name="Stage",
                value_name="Value"
            )
        )

        fig_compare = px.bar(
            compare_long,
            x="Metric",
            y="Value",
            color="Stage",
            barmode="group",
            title="Before vs After Preprocessing"
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )


        st.subheader(
            "Preprocessing Report"
        )

        report_df = pd.DataFrame(
            report.items(),
            columns=[
                "Operation",
                "Result"
            ]
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )


        st.subheader(
            "Cleaned Dataset Preview"
        )

        st.dataframe(
            cleaned.head(100),
            use_container_width=True
        )


# ============================================================
# VISUAL ANALYTICS
# ============================================================

with tabs[4]:

    st.header(
        "Interactive Visual Analytics"
    )

    active_df = (
        st.session_state.get(
            "cleaned_df",
            df
        )
    )

    active_numeric = (
        active_df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    active_categorical = (
        active_df
        .select_dtypes(
            exclude=np.number
        )
        .columns
        .tolist()
    )


    if active_numeric:

        numeric_feature = st.selectbox(
            "Numeric feature",
            active_numeric,
            key="visual_numeric"
        )

        fig = px.histogram(
            active_df,
            x=numeric_feature,
            marginal="box",
            nbins=40,
            title=f"Distribution of {numeric_feature}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    if active_categorical:

        category = st.selectbox(
            "Categorical feature",
            active_categorical,
            key="visual_category"
        )

        counts = (
            active_df[category]
            .astype(str)
            .value_counts()
            .head(20)
            .reset_index()
        )

        counts.columns = [
            category,
            "Count"
        ]

        fig_category = px.bar(
            counts,
            x=category,
            y="Count",
            title=f"Top Categories: {category}"
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )


    if len(active_numeric) >= 2:

        st.subheader(
            "Interactive Scatter Plot"
        )

        s1, s2 = st.columns(2)

        with s1:

            x_feature = st.selectbox(
                "X-axis",
                active_numeric,
                index=0
            )

        with s2:

            y_feature = st.selectbox(
                "Y-axis",
                active_numeric,
                index=1
            )

        color_options = (
            ["None"]
            + active_df.columns.tolist()
        )

        color_feature = st.selectbox(
            "Color grouping",
            color_options
        )

        color_value = (
            None
            if color_feature == "None"
            else color_feature
        )

        scatter = px.scatter(
            active_df,
            x=x_feature,
            y=y_feature,
            color=color_value,
            title=(
                f"{x_feature} vs {y_feature}"
            )
        )

        st.plotly_chart(
            scatter,
            use_container_width=True
        )


# ============================================================
# CORRELATIONS
# ============================================================

with tabs[5]:

    st.header(
        "Correlation Analysis"
    )

    active_df = (
        st.session_state.get(
            "cleaned_df",
            df
        )
    )

    numeric_df = (
        active_df
        .select_dtypes(
            include=np.number
        )
    )

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )


        upper = corr.where(
            np.triu(
                np.ones(
                    corr.shape
                ),
                k=1
            ).astype(bool)
        )

        pairs = (
            upper
            .stack()
            .reset_index()
        )

        pairs.columns = [
            "Feature 1",
            "Feature 2",
            "Correlation"
        ]

        pairs[
            "Absolute Correlation"
        ] = (
            pairs[
                "Correlation"
            ].abs()
        )

        pairs = pairs.sort_values(
            "Absolute Correlation",
            ascending=False
        )

        st.subheader(
            "Strongest Relationships"
        )

        st.dataframe(
            pairs.head(20),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "At least two numeric columns are required."
        )


# ============================================================
# FILTER
# ============================================================

with tabs[6]:

    st.header(
        "Interactive Data Filter"
    )

    active_df = (
        st.session_state.get(
            "cleaned_df",
            df
        )
    )

    filter_column = st.selectbox(
        "Select column",
        active_df.columns,
        key="filter_column"
    )

    filtered = active_df.copy()


    if pd.api.types.is_numeric_dtype(
        active_df[filter_column]
    ):

        valid_values = (
            active_df[
                filter_column
            ]
            .dropna()
        )

        if not valid_values.empty:

            minimum = float(
                valid_values.min()
            )

            maximum = float(
                valid_values.max()
            )

            if minimum < maximum:

                selected_range = st.slider(
                    "Select range",
                    minimum,
                    maximum,
                    (
                        minimum,
                        maximum
                    )
                )

                filtered = active_df[
                    active_df[
                        filter_column
                    ].between(
                        selected_range[0],
                        selected_range[1]
                    )
                ]

    else:

        options = sorted(
            active_df[
                filter_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_values = (
            st.multiselect(
                "Select values",
                options,
                default=options
            )
        )

        filtered = active_df[
            active_df[
                filter_column
            ]
            .astype(str)
            .isin(
                selected_values
            )
        ]


    st.metric(
        "Filtered Rows",
        f"{len(filtered):,}"
    )

    st.dataframe(
        filtered,
        use_container_width=True
    )


# ============================================================
# EXPORT
# ============================================================

with tabs[7]:

    st.header(
        "Export Prepared Dataset"
    )

    if "cleaned_df" in st.session_state:

        export_df = (
            st.session_state[
                "cleaned_df"
            ]
        )

        st.success(
            "The cleaned dataset is ready for export."
        )

    else:

        export_df = df

        st.info(
            """
            Data cleaning has not been run yet.
            The original dataset will be exported.
            """
        )


    csv = export_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "⬇ Download Prepared CSV",
        data=csv,
        file_name=(
            "masmi_prepared_dataset.csv"
        ),
        mime="text/csv",
        type="primary",
        use_container_width=True
    )


    st.subheader(
        "Export Summary"
    )

    st.write(
        f"Rows: **{len(export_df):,}**"
    )

    st.write(
        f"Columns: **{len(export_df.columns):,}**"
    )

    st.write(
        "Format: **CSV**"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:20px;
        opacity:.75;
    ">

    <b>Smart Data Analytics & Preprocessing Dashboard</b><br>

    Developed by A. Masmi • jovina@gmx.us<br>

    Python • pandas • NumPy • SciPy • Plotly • Streamlit<br>

    Session: {session_date} • {session_time}

    </div>
    """,
    unsafe_allow_html=True
)
