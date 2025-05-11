import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def plot_gain_lift_plotly(
    df_spark: DataFrame,
    label_col: str = "label",
    probability_col: str = "probability",
    bins: int = 10
) -> go.Figure:
    """
    Efficient Spark-based Gain and Lift charts using ntile for decile bucketing.

    Parameters:
        df_spark: Input Spark DataFrame.
        label_col (str): Column name for true binary labels (0/1).
        probability_col (str): Column name for prediction probability of positive class.
        bins (int): Number of quantile buckets (e.g., 10 for deciles).

    Returns:
        go.Figure: Plotly figure with Gain and Lift subplots.
    """
    # 1) Assign buckets via ntile over descending probability
    bucket_window = Window.orderBy(F.col(probability_col).desc())
    df = df_spark.withColumn("bucket", F.ntile(bins).over(bucket_window))

    # 2) Aggregate per bucket: count positives and population
    stats = (
        df.groupBy("bucket")
          .agg(
              F.sum(F.col(label_col)).alias("bucket_positives"),
              F.count(F.col(label_col)).alias("bucket_size")
          )
          .orderBy("bucket")
    )

    # 3) Compute cumulative positives
    cum_window = Window.orderBy("bucket").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    stats = stats.withColumn("cum_positives", F.sum("bucket_positives").over(cum_window))

    # 4) Total positives for gain denominator
    total_pos = df_spark.select(F.sum(F.col(label_col))).first()[0] or 0

    # 5) Compute gain, population percent, and lift
    stats = (
        stats
          .withColumn("gain", F.col("cum_positives") / F.lit(total_pos))
          .withColumn("pop_pct", F.col("bucket") / F.lit(bins))
          .withColumn("lift", F.when(F.col("pop_pct") > 0, F.col("gain") / F.col("pop_pct")).otherwise(0))
    )

    # 6) Convert to Pandas for plotting
    pd_stats = (
        stats.select("bucket", "bucket_size", "bucket_positives", "cum_positives", "gain", "pop_pct", "lift")
             .toPandas()
             .sort_values("bucket")
    )

    # 7) Build subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Gain Chart", "Lift Chart"),
        horizontal_spacing=0.15
    )

    # Gain plugin
    fig.add_trace(
        go.Scatter(
            x=pd_stats["pop_pct"], y=pd_stats["gain"],
            mode="lines+markers", name="Model Gain",
            hovertemplate="Population: %{x:.0%}<br>Gain: %{y:.0%}<extra></extra>"
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), showlegend=False, hoverinfo="skip"
        ), row=1, col=1
    )
    fig.add_annotation(
        x=0.55, y=0.5, xref="x1", yref="y1",
        text="Baseline (Random model)", showarrow=False,
        font=dict(color="gray", size=12), textangle=-43
    )

    # Lift plugin
    fig.add_trace(
        go.Scatter(
            x=pd_stats["pop_pct"], y=pd_stats["lift"],
            mode="lines+markers", name="Model Lift",
            hovertemplate="Population: %{x:.0%}<br>Lift: %{y:.2f}<extra></extra>"
        ), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[1, 1], mode="lines",
            line=dict(dash="dash", color="gray"), showlegend=False, hoverinfo="skip"
        ), row=1, col=2
    )
    fig.add_annotation(
        x=0.5, y=1.05, xref="x2", yref="y2",
        text="Baseline (Lift=1)", showarrow=False,
        font=dict(color="gray", size=12)
    )

    # 8) Axes formatting
    fig.update_xaxes(title="Cumulative Population %", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title="Cumulative Positives %", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title="Cumulative Population %", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title="Lift (Gain / Cum. Pop. %)", row=1, col=2)

    # 9) Layout
    fig.update_layout(
        legend=dict(x=0.35, y=1.15, orientation="h"),
        width=900, height=500, template="plotly_white"
    )

    return fig