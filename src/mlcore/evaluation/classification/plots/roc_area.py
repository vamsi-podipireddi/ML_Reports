# mlcore/evaluation/classification/plots/roc_area.py
"""
Overlayed ROC area plot for train and test datasets.
"""
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def plot_roc_curve_area(
    train_eval_df: pd.DataFrame,
    test_eval_df: pd.DataFrame
) -> go.Figure:
    """
    Overlay smooth ROC areas for train and test, hide their legend entries,
    and place the AUC values inside the plot:
      - if AUC > 0.5 → bottom-right,
      - if AUC < 0.5 → top-left.

    Parameters:
        train_eval_df (pd.DataFrame): DataFrame with columns ["threshold","TP","TN","FP","FN"].
        test_eval_df  (pd.DataFrame): Same format for test set.

    Returns:
        go.Figure: Plotly Figure of overlaid ROC areas.
    """
    def prepare(df: pd.DataFrame):
        dfc = df.copy()
        dfc["fpr"] = dfc["FP"] / (dfc["FP"] + dfc["TN"])
        dfc["tpr"] = dfc["TP"] / (dfc["TP"] + dfc["FN"])
        dfc = dfc.sort_values("fpr")
        auc = np.trapz(dfc["tpr"].values, dfc["fpr"].values)
        return dfc, auc

    def pos_for_auc(auc: float):
        if auc > 0.5:
            return dict(x=0.95, y=0.05, xanchor="right", yanchor="bottom")
        else:
            return dict(x=0.05, y=0.95, xanchor="left",  yanchor="top")

    df_train, auc_train = prepare(train_eval_df)
    df_test,  auc_test  = prepare(test_eval_df)

    fig = go.Figure()

    # ---- Train ROC ----
    fig.add_trace(go.Scatter(
        x=df_train["fpr"], y=df_train["tpr"],
        mode="lines", name="Train ROC (hidden)",
        hovertemplate=(
            "Train ROC<br>"
            "Threshold %{text}<br>"
            "FPR = %{x:.2f}<br>"
            "TPR = %{y:.2f}<extra></extra>"
        ),
        text=[f"{t:.2f}" for t in df_train["threshold"]],
        line=dict(color="rgb(31,119,180)", shape="spline", smoothing=1.3),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
        showlegend=False
    ))

    # ---- Test ROC ----
    fig.add_trace(go.Scatter(
        x=df_test["fpr"], y=df_test["tpr"],
        mode="lines", name="Test ROC (hidden)",
        hovertemplate=(
            "Test ROC<br>"
            "Threshold %{text}<br>"
            "FPR = %{x:.2f}<br>"
            "TPR = %{y:.2f}<extra></extra>"
        ),
        text=[f"{t:.2f}" for t in df_test["threshold"]],
        line=dict(color="rgb(255,127,14)", shape="spline", smoothing=1.3),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.2)",
        showlegend=False
    ))

    # diagonal chance line
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray")
    )

    # annotations
    p_train = pos_for_auc(auc_train)
    fig.add_annotation(
        text=f"<b>Train AUC:</b> {auc_train:.2f}",
        showarrow=False,
        font=dict(color="rgb(31,119,180)", size=14),
        bgcolor="rgba(255,255,255,0.7)",
        xref="paper", yref="paper", **p_train
    )

    p_test = pos_for_auc(auc_test)
    if p_test == p_train:
        p_test['y'] += 0.07
    fig.add_annotation(
        text=f"<b>Test AUC:</b> {auc_test:.2f}",
        showarrow=False,
        font=dict(color="rgb(255,127,14)", size=14),
        bgcolor="rgba(255,255,255,0.7)",
        xref="paper", yref="paper", **p_test
    )

    fig.update_layout(
        title={"text": "ROC Curves", "x": 0.5},
        xaxis_title="False Positive Rate (1 – Specificity)",
        yaxis_title="True Positive Rate (Recall)",
        xaxis=dict(range=[0,1], showgrid=True),
        yaxis=dict(range=[0,1], showgrid=True),
        plot_bgcolor="white", paper_bgcolor="white",
        width=700, height=600
    )

    return fig
