import numpy as np
import plotly.graph_objects as go

# ─── Helpers ────────────────────────────────────────────────────────────────

def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Simple trapezoidal AUC."""
    return np.trapz(y, x)

def make_zone_polygons(random_fpr: np.ndarray, random_tpr: np.ndarray):
    """Return two dicts of kwargs for the red (below) and green (above) fill polygons."""
    red = dict(
        x=np.concatenate([random_fpr, random_fpr[::-1]]),
        y=np.concatenate([random_tpr, np.zeros_like(random_tpr)]),
        fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    )
    green = dict(
        x=np.concatenate([random_fpr, random_fpr[::-1]]),
        y=np.concatenate([random_tpr, np.ones_like(random_tpr)]),
        fill='toself', fillcolor='rgba(0,128,0,0.2)', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    )
    return red, green

# ─── Main plotting function ─────────────────────────────────────────────────

def plot_reference_rocs():
    # 1) Define random diagonal
    random_fpr = np.linspace(0, 1, 200)
    random_tpr = random_fpr

    # 2) Define curves and compute their AUCs
    curves = [
        {
            "name": "Random",
            "x": random_fpr, "y": random_tpr,
            "line": dict(dash='dash', color='black', width=2),
            "annot_x": 0.4, "annot_y": 0.4,
            "annot_offset": (0.4, 0.55),
            "annot_color": "black",
            "annot_text": "AUC=0.50 (Random)",
            "textangle": 0
        },
        {
            "name": "Perfect",
            "x": np.array([0, 0, 1]), "y": np.array([0, 1, 1]),
            "line": dict(color='green', width=3),
            "annot_x": 0.2, "annot_y": 1.0,
            "annot_offset": (0.2, 1.1),
            "annot_color": "green",
            "annot_text": "AUC=1.00 (Perfect)",
            "textangle": 0
        },
        {
            "name": "Good",
            "x": np.linspace(0, 1, 200),
            "y": None,  # to be filled below
            "line": dict(color='blue', width=3),
            "annot_x": 0.3, "annot_y": None,
            "annot_offset": (0.3, 0.8),
            "annot_color": "blue",
            "annot_text": "AUC ∈ (0.5, 1) (Good)",
            "textangle": 0
        },
        {
            "name": "Poor",
            "x": np.linspace(0, 1, 200),
            "y": None,
            "line": dict(color='red', width=3),
            "annot_x": 0.5, "annot_y": None,
            "annot_offset": (0.5, 0.3),
            "annot_color": "red",
            "annot_text": " AUC ∈ (0, 0.5) (Poor)",
            "textangle": 0
        }
    ]

    # Populate Good & Poor curves with data + AUC + annotation props
    for c in curves:
        if c["name"] == "Good":
            c["y"] = np.cbrt(c["x"])
            auc = compute_auc(c["x"], c["y"])
            c["annot_y"] = np.cbrt(c["annot_x"])
        elif c["name"] == "Poor":
            c["y"] = c["x"]**3
            auc = compute_auc(c["x"], c["y"])
            c["annot_y"] = c["annot_x"]**3

    # 3) Build figure
    fig = go.Figure()

    # 3a) Add zones
    red_zone, green_zone = make_zone_polygons(random_fpr, random_tpr)
    fig.add_trace(go.Scatter(**red_zone))
    fig.add_trace(go.Scatter(**green_zone))

    # 3b) Add all curves
    for c in curves:
        fig.add_trace(go.Scatter(
            x=c["x"], y=c["y"], mode='lines',
            line=c["line"],
            showlegend=False
        ))

    # 4) Add annotations
    for c in curves:
        fig.add_annotation(
            x=c["annot_x"], y=c["annot_y"],
            ax=c["annot_offset"][0], ay=c["annot_offset"][1],
            xref="x", yref="y", axref="x", ayref="y",
            text=c["annot_text"],
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor=c["annot_color"], arrowwidth=1,
            textangle=c["textangle"],
            font=dict(color=c["annot_color"], size=14)
        )

    # 5) Final layout
    fig.update_layout(
        title=dict(
            text="Reference ROC Curve",
            x=0.5,  # Center the title
            xanchor='center'
        ),
        xaxis_title="False Positive Rate (1 – Specificity)",
        yaxis_title="True Positive Rate (Recall)",
        plot_bgcolor="white", paper_bgcolor="white",
        width=700, height=600
    )

    return fig

