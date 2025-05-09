import numpy as np
import plotly.graph_objects as go

def plot_reference_pr_curves() -> go.Figure:
    """
    Reference Precision–Recall curves illustrating Perfect, Good, and Poor behavior
    with colored bands and on‐curve annotations.
    """
    p = np.linspace(0, 1, 200)
    perfect_r = np.ones_like(p)
    L = 0.5
    good_r = 1 - (1 - L) * p**4
    auc_good = np.trapz(good_r, p)

    fig = go.Figure()

    # Colored bands
    bands = [
        (0.5, 1.0, "rgba(0,255,0,0.2)"),   # ≥0.5: light green
        (0.25, 0.5, "rgba(255,0,0,0.1)"),  # 0.25–0.5: light red
        (0.0, 0.25, "rgba(255,0,0,0.3)"),  # <0.25: darker red
    ]
    for y0, y1, color in bands:
        fig.add_shape(dict(
            type="rect",
            xref="paper", x0=0, x1=1,
            yref="y", y0=y0, y1=y1,
            fillcolor=color, line=dict(width=0), layer="below"
        ))

    # Baseline lines
    for baseline in (0.5, 0.25):
        fig.add_shape(dict(
            type="line",
            x0=0, y0=baseline, x1=1, y1=baseline,
            line=dict(dash="dash", color="black"), layer="above"
        ))

    # PR curves
    fig.add_trace(go.Scatter(
        x=p, y=perfect_r, mode="lines",
        line=dict(color="green", width=3), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=p, y=good_r, mode="lines",
        line=dict(color="blue", width=3, shape="spline", smoothing=1.2),
        showlegend=False
    ))

    # Annotations
    annotations = [
        (0.85, 0.95, "AP = 1.00 (Perfect)", "green"),
        (0.75, 0.70, f"AP ≈ {auc_good:.2f} (Good)", "blue"),
        (0.15, 0.50, "Baseline (P:N=1:1)", "black"),
        (0.15, 0.25, "Baseline (P:N=1:3)", "black"),
        (0.75, 0.35, "Poor (P:N=1:1)", "rgba(255,0,0,0.5)"),
        (0.75, 0.15, "Poor (P:N=1:3)", "rgba(255,0,0,0.7)"),
    ]
    for x, y, text, color in annotations:
        fig.add_annotation(
            x=x, y=y, text=text, font=dict(color=color, size=14),
            showarrow=False, yshift=10
        )

    # Axis styling
    fig.update_xaxes(
        title="Recall", range=[0,1],
        showgrid=False, zeroline=False, showline=True,
        linecolor="black", linewidth=2
    )
    fig.update_yaxes(
        title="Precision", range=[0,1],
        showgrid=False, zeroline=False, showline=True,
        linecolor="black", linewidth=2
    )

    fig.update_layout(
        title={"text": "Reference PR Curve", "x": 0.5, "xanchor": "center"},
        plot_bgcolor="white", paper_bgcolor="white",
        width=700, height=600
    )

    return fig
