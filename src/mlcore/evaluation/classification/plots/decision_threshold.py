import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_decision_threshold_slider(
    df_metrics: pd.DataFrame,
    metrics: list = ["precision", "recall", "f1", "accuracy", "specificity"],
    default_threshold: float = 0.5
) -> go.Figure:
    """
    Plot several metrics vs. threshold with a slider.
    Only precision & recall are shown initially; the rest start as 'legendonly'.
    Legend‐click will toggle any curve on/off.
    """
    # 1) Sort and find default index
    df = df_metrics.sort_values("threshold").reset_index(drop=True)
    if default_threshold in df["threshold"].values:
        default_idx = int(df.index[df["threshold"] == default_threshold][0])
    else:
        default_idx = 0

    # 2) Create the figure and metric traces
    colors = {
        "precision":   "blue",
        "recall":      "red",
        "f1":          "green",
        "accuracy":    "purple",
        "specificity": "orange",
        "npv":         "brown",
        "balanced_accuracy": "pink",
        "jaccard":     "gray",
        "gmean":       "cyan",
        "hamming_loss": "magenta",
    }
    default_on = {"precision", "recall"}

    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=df["threshold"],
            y=df[metric],
            mode="lines",
            name=metric.capitalize(),
            line=dict(color=colors[metric], shape="spline", smoothing=1.3),
            marker=dict(size=6),
            visible=True if metric in default_on else "legendonly",
            legendgroup=metric.capitalize()
        ))

    # 3) Vertical threshold line
    t0 = df.loc[default_idx, "threshold"]
    fig.add_shape(dict(
        type="line", x0=t0, x1=t0, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(dash="dot", color="black", width=2)
    ))

    # 4) Annotation box showing all metrics at current threshold
    def make_text(idx):
        lines = []
        for m in metrics:
            val = df.loc[idx, m]
            lines.append(f"<b>{m.capitalize()}:</b> {val:.2f}")
        return "<br>".join(lines)

    fig.update_layout(
        annotations=[dict(
            x=1.02, y=0.02, xref="paper", yref="paper",
            text=make_text(default_idx),
            showarrow=False, align="left",
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            opacity=0.8
        )],
        margin=dict(l=60, r=160, t=80, b=80)
    )

    # 5) Slider steps: move the line & update the annotation
    steps = []
    for i, row in df.iterrows():
        t = row["threshold"]
        args = {
            "shapes[0].x0": t,
            "shapes[0].x1": t,
            "annotations[0].text": make_text(i)
        }
        steps.append(dict(
            method="relayout",
            label=f"{t:.2f}",
            args=[args]
        ))

    slider = {
        "active": default_idx,
        "pad": {"t": 50},
        "len": 0.8,
        "x": 0.1,
        "steps": steps,
        "currentvalue": {
            "visible": True,
            "prefix": "Threshold: ",
            "xanchor": "center",
            "font": {"size": 14, "color": "black"},
        },
        "ticklen":   0,
        "tickwidth": 0,
        "tickcolor": "rgba(0,0,0,0)",
        "font": {"color": "rgba(0,0,0,0)"},
        "bgcolor": "white",
        "bordercolor": "lightgray",
    }

    # 6) Final layout tweaks
    fig.update_layout(
        sliders=[slider],
        title="Decision Threshold Metrics\n(Precision & Recall shown by default)",
        xaxis=dict(
            title="Threshold", range=[0,1], dtick=0.1,
            showgrid=True, gridcolor="lightgray"
        ),
        yaxis=dict(
            title="Metric Value", range=[0,1], dtick=0.1,
            showgrid=True, gridcolor="lightgray"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=900, height=600,
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )

    return fig
