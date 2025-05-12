# mlcore/evaluation/classification/plots/calibration.py

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import plotly.graph_objects as go

def plot_calibration_curve_compare(
    train_df: DataFrame,
    test_df: DataFrame,
    label_col: str = "label",
    probability_col: str = "probability",
    bins: int = 10
) -> go.Figure:
    """
    Compare train & test calibration (reliability) diagrams in one figure.
    
    Bins predicted probs into `bins` equal-width intervals,
    computes per-bin mean predicted probability and actual positive rate
    for both datasets, then overlays both curves plus the y=x line.
    Legend at bottom center.
    """
    def _compute_stats(df_spark):
        # assign bucket 0..bins-1
        df = df_spark.withColumn(
            "bucket",
            F.least(
                F.floor(F.col(probability_col) * bins).cast("int"),
                F.lit(bins - 1)
            )
        )
        # aggregate
        stats = (
            df.groupBy("bucket")
              .agg(
                  F.avg(F.col(probability_col)).alias("avg_pred"),
                  (F.sum(F.col(label_col)) / F.count("*")).alias("frac_pos")
              )
              .orderBy("bucket")
        )
        # center of each bin
        stats = stats.withColumn(
            "bin_center",
            (F.col("bucket") + F.lit(0.5)) / F.lit(bins)
        )
        # to pandas
        return stats.select("bin_center", "avg_pred", "frac_pos") \
                    .toPandas() \
                    .sort_values("bin_center")

    # compute pandas stats
    pdf_train = _compute_stats(train_df)
    pdf_test  = _compute_stats(test_df)

    fig = go.Figure()

    # perfect diagonal
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfectly calibrated",
        hoverinfo="skip"
    ))

    # train
    fig.add_trace(go.Scatter(
        x=pdf_train["bin_center"],
        y=pdf_train["frac_pos"],
        mode="lines+markers",
        name="Train",
        line=dict(color="rgb(31,119,180)"),
        marker=dict(symbol="circle"),
        hovertemplate=(
            "Train<br>Bin center: %{x:.2f}<br>"
            "Avg pred: %{customdata[0]:.2f}<br>"
            "Observed: %{y:.2f}<extra></extra>"
        ),
        customdata=pdf_train[["avg_pred"]].values
    ))

    # test (hidden by default)
    fig.add_trace(go.Scatter(
        x=pdf_test["bin_center"],
        y=pdf_test["frac_pos"],
        mode="lines+markers",
        name="Test",
        line=dict(color="rgb(255,127,14)"),
        marker=dict(symbol="square"),
        hovertemplate=(
            "Test<br>Bin center: %{x:.2f}<br>"
            "Avg pred: %{customdata[0]:.2f}<br>"
            "Observed: %{y:.2f}<extra></extra>"
        ),
        customdata=pdf_test[["avg_pred"]].values,
        visible="legendonly"
    ))

    # layout & legend
    fig.update_layout(
        title={
            "text": "Calibration Curve (Reliability Diagram)",
            "x": 0.5, "xanchor": "center"
        },
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed positive rate",
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[0,1]),
        width=750, height=550,
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0.50, xanchor="center",
            y=1.1, yanchor="top"
        )
    )

    return fig
