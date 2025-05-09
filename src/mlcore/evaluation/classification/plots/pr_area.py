import numpy as np
import plotly.graph_objects as go
import pandas as pd

def plot_pr_curve_area(
    train_eval_df: pd.DataFrame,
    test_eval_df: pd.DataFrame
) -> go.Figure:
    """
    Overlay smooth Precision–Recall areas for train and test,
    hide their own legend entries, place the Average Precision (AP)
    and also draw horizontal baseline lines at P/(P+N) for each set,
    labeling them with the P:N ratio.
    """
    def prepare(dfm: pd.DataFrame):
        # Compute P, N, baseline and P:N ratio
        first = dfm.iloc[0]
        P, N = first.TP + first.FN, first.TN + first.FP
        baseline = P / (P + N)
        pn_ratio = P / N

        # Build PR table with endpoint
        dfc = dfm[["threshold", "recall", "precision"]].copy()
        endpoint = pd.DataFrame({
            "threshold": [1.0], "recall": [0.0], "precision": [1.0]
        })
        dfc = (
            pd.concat([dfc, endpoint], ignore_index=True)
              .drop_duplicates("threshold", keep="last")
              .sort_values("recall")
              .reset_index(drop=True)
        )
        ap_val = np.trapz(dfc["precision"], dfc["recall"])
        return dfc, ap_val, baseline, pn_ratio

    def corner_for_ap(ap: float):
        return dict(
            x=0.4, y=0.05, xanchor="left", yanchor="bottom"
        ) if ap > 0.5 else dict(
            x=0.75, y=0.95, xanchor="right", yanchor="top"
        )

    # Prepare both sets
    df_train, ap_train, base_train, pn_train = prepare(train_eval_df)
    df_test,  ap_test,  base_test,  pn_test  = prepare(test_eval_df)

    fig = go.Figure()

    # Train PR area
    # --- Train PR curve (no legend) ---
    fig.add_trace(go.Scatter(
        x=df_train["recall"], y=df_train["precision"],
        mode="lines",
        line=dict(color="rgb(31,119,180)", shape="spline", smoothing=1.3),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
        marker=dict(size=6),
        showlegend=False,
        hovertemplate=(
            "Train PR<br>"
            "Threshold %{text}<br>"
            "Recall = %{x:.2f}<br>"
            "Precision = %{y:.2f}<extra></extra>"
        ),
        text=[f"{t:.2f}" for t in df_train["threshold"]]
    ))

    # --- Test PR curve (no legend) ---
    fig.add_trace(go.Scatter(
        x=df_test["recall"], y=df_test["precision"],
        mode="lines",
        line=dict(color="rgb(255,127,14)", shape="spline", smoothing=1.3),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.2)",
        marker=dict(size=6),
        showlegend=False,
        hovertemplate=(
            "Test PR<br>"
            "Threshold %{text}<br>"
            "Recall = %{x:.2f}<br>"
            "Precision = %{y:.2f}<extra></extra>"
        ),
        text=[f"{t:.2f}" for t in df_test["threshold"]]
    ))

    # --- Baseline lines (with legend entries) ---
    # Train baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[base_train, base_train],
        mode="lines",
        line=dict(dash="dot", color="rgb(31,119,180)", width=2),
        name=f"Train baseline (P:N≈{pn_train:.2f}:1)",
        showlegend=False,
    ))
    # Test baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[base_test,  base_test],
        mode="lines",
        line=dict(dash="dash", color="rgb(255,127,14)", width=2),
        name=f"Test baseline (P:N≈{pn_test:.2f}:1)",
        showlegend=False,
    ))

    # ---- AUC annotations (corner) ----
    p_train = corner_for_ap(ap_train)
    fig.add_annotation(
        text=f"<b>Train AP:</b> {ap_train:.2f}",
        showarrow=False,
        font=dict(color="rgb(31,119,180)", size=14),
        bgcolor="rgba(255,255,255,0.7)",
        xref="paper", yref="paper",
        **p_train
    )

    p_test = corner_for_ap(ap_test)
    # if they’d collide, shift only the TEST one vertically:
    if p_test == p_train:
        p_test["y"] += 0.07 if p_test["yanchor"]=="bottom" else -0.07

    fig.add_annotation(
        text=f"<b>Test AP:</b> {ap_test:.2f}",
        showarrow=False,
        font=dict(color="rgb(255,127,14)", size=14),
        bgcolor="rgba(255,255,255,0.7)",
        xref="paper", yref="paper",
        **p_test
    )

    # ---- Baseline annotations (at end of line) ----
    fig.add_annotation(
        x=0.3, y=base_train - 0.05,
        xref="x", yref="y",
        text=f"Train baseline (P:N≈{pn_train:.1f}:1.0)",
        showarrow=False,
        font=dict(color="rgb(31,119,180)", size=12),
        xanchor="right", yanchor="middle",
        # bgcolor="rgba(255,255,255,0.7)"
    )
    fig.add_annotation(
        x=1, y=base_test+0.05,
        xref="x", yref="y",
        text=f"Test baseline (P:N≈{pn_test:.1f}:1.0)",
        showarrow=False,
        font=dict(color="rgb(255,127,14)", size=12),
        xanchor="right", yanchor="middle",
        # bgcolor="rgba(255,255,255,0.7)"
    )

    # --- Layout ---
    fig.update_layout(
        title={
            "text": "Train vs. Test Precision–Recall Curves",
            "x": 0.5, "xanchor": "center"
        },
        title_x=0.5,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0,1], showgrid=True),
        yaxis=dict(range=[0,1], showgrid=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=750, height=600
    )

    return fig
