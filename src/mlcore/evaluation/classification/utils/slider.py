# mlcore/evaluation/classification/utils/slider.py

from typing import List, Dict
import pandas as pd

from .color import metric_color, METRICS_CONFIG

# Labels for confusion‐matrix heatmap axes
CM_X = ["Predicted '0'", "Predicted '1'"]
CM_Y = ["Actual '0'",    "Actual '1'"]

def create_slider_steps(
    df_metrics: pd.DataFrame,
    metrics: List[str]
) -> List[Dict]:
    """
    Generate Plotly slider steps to update both the confusion matrix and metrics bar chart.

    Parameters:
        df_metrics (pd.DataFrame): Each row contains TN, FP, FN, TP and metric values.
        metrics (List[str]): Names of metrics (columns in df_metrics) to display.

    Returns:
        List[Dict]: List of dicts, one per threshold, for Plotly slider 'steps'.
    """
    steps: List[Dict] = []

    for _, r in df_metrics.iterrows():
        # Build confusion matrix counts
        z = [[r.TN, r.FP], [r.FN, r.TP]]
        # Metric values in same order
        vals = [r[m] for m in metrics]
        # Color each bar according to its value
        bar_colors = [metric_color(r[m], METRICS_CONFIG[m]["invert"]) for m in metrics]
        # Text labels for bars
        bar_text = [f"{v:.2f}" for v in vals]

        steps.append({
            'method': 'update',
            'args': [
                {
                    # first trace = heatmap, second = bar chart
                    'z': [z, None],
                    'x': [CM_X, vals],
                    'marker.color': [None, bar_colors],
                    'text': [z, bar_text]
                },
                {'transition': {'duration': 500, 'easing': 'linear'}}
            ],
            'label': f"{r.threshold:.2f}"
        })

    return steps

def configure_slider(
    steps: List[Dict],
    active_index: int
) -> Dict:
    """
    Build the Plotly slider configuration dict.

    Parameters:
        steps (List[Dict]): Slider steps from create_slider_steps().
        active_index (int): Index of the default active threshold.

    Returns:
        Dict: Slider configuration for fig.update_layout(sliders=[...]).
    """
    return {
        "active": active_index,
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
        # hide ticks
        "ticklen":   0,
        "tickwidth": 0,
        "tickcolor": "rgba(0,0,0,0)",
        "font":     {"color": "rgba(0,0,0,0)"},
        "bgcolor":   "white",
        "bordercolor": "lightgray",
    }
