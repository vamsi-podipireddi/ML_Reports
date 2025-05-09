# mlcore/evaluation/classification/plots/interactive.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mlcore.evaluation.classification.metrics import compute_threshold_metrics
from mlcore.evaluation.classification.utils.color import metric_color, METRICS_CONFIG
from mlcore.evaluation.classification.utils.slider import (
    create_slider_steps, configure_slider
)

CM_X = ["Predicted '0'", "Predicted '1'"]
CM_Y = ["Actual '0'", "Actual '1'"]

def build_confusion_metrics_figure(df_metrics, metrics, default_idx):
    # extract initial values
    init = df_metrics.iloc[default_idx]
    initial_z = [[init.TN, init.FP], [init.FN, init.TP]]
    initial_vals = [init[m] for m in metrics]

    # build figure
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Confusion Matrix", "Metrics")
    )

    # confusion matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=initial_z, x=CM_X, y=CM_Y,
            text=initial_z, texttemplate="%{text}", showscale=False,
            colorscale="Greens"
        ),
        row=1, col=1
    )

    # metrics bar chart
    bar_colors = [
        metric_color(val, METRICS_CONFIG[m]["invert"])
        for val, m in zip(initial_vals, metrics)
    ]
    fig.add_trace(
        go.Bar(
            y=metrics, x=initial_vals, orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in initial_vals], textposition='auto'
        ),
        row=1, col=2
    )

    # sliders
    steps = create_slider_steps(df_metrics, metrics)
    slider = configure_slider(steps, default_idx)
    # Update global layout attributes
    fig.update_layout(
        sliders=[slider],
        transition={'duration': 500, 'easing': 'linear'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'sans-serif', 'size': 12, 'color': '#333'}
    )
    # Configure x and y axes for both subplots
    fig.update_xaxes(showgrid=False, ticklabelstandoff=10, tickfont=dict(size=12, family='sans-serif'))
    fig.update_yaxes(showgrid=False, ticklabelstandoff=10, tickfont=dict(size=12, family='sans-serif'))
    fig.update_xaxes(tickmode='array', tickvals=CM_X, type='category', row=1, col=1)
    fig.update_yaxes(tickmode='array', tickvals=CM_Y, type='category', row=1, col=1)
    fig.update_xaxes(autorange=False, showticklabels=False, range=[1, 0], row=1, col=2)
    fig.update_yaxes(side='right', row=1, col=2)

    # Enhance subplot titles appearance
    for ann in fig.layout.annotations:
        ann.text = f"<b>{ann.text}</b>"
        ann.y += 0.05
        ann.font = {'family': 'sans-serif', 'size': 16, 'color': 'DarkSlateGray'}
    return fig
