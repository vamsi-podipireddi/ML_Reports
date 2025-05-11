# mlcore/evaluation/classification/evaluator.py

from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame
from mlcore.evaluation.classification.metrics import compute_threshold_metrics
from mlcore.evaluation.classification.plots import (
    build_confusion_metrics_figure,
    plot_reference_rocs,
    plot_roc_curve_area,
    plot_reference_pr_curves,
    plot_pr_curve_area,
    plot_decision_threshold_slider,
    plot_gain_lift_plotly,
)


class BinaryClassifierEvaluator:
    """Evaluate binary classifiers with thresholded metrics and interactive plots."""

    def __init__(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        label_col: str = "label",
        probability_col: str = "probability",
        thresholds: Optional[List[float]] = None,
        default_threshold: float = 0.5,
        extra_metrics: bool = False,
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.label_col = label_col
        self.probability_col = probability_col
        self.thresholds = thresholds or [i / 20 for i in range(21)]
        self.default_threshold = default_threshold
        self.extra_metrics = extra_metrics

        # internal caches
        self._metrics_dfs: Dict[str, Any] = {}
        self.plots: Optional[Dict[str, Any]] = None

        try:
            self.default_idx = self.thresholds.index(default_threshold)
        except ValueError:
            self.default_idx = 0

    def __metrics_to_calculate(self) -> List[str]:
        base = ["accuracy", "precision", "recall", "f1"]
        extra = (
            [
                "specificity",
                "npv",
                "balanced_accuracy",
                "jaccard",
                "gmean",
                "hamming_loss",
            ]
            if self.extra_metrics
            else []
        )
        return base + extra

    def _compute_for_df(self, df: DataFrame, key: str) -> Any:
        """Compute & cache the full threshold-metrics DataFrame for a given split."""
        if key not in self._metrics_dfs:
            self._metrics_dfs[key] = compute_threshold_metrics(
                df_spark=df,
                label_col=self.label_col,
                probability_col=self.probability_col,
                thresholds=self.thresholds,
                extra_metrics=self.extra_metrics,
            )
        return self._metrics_dfs[key]

    def _select_row(self, df_metrics, threshold: float) -> Dict[str, float]:
        """Validate & fetch the metrics dict for a single threshold from one DataFrame."""
        if threshold not in df_metrics["threshold"].values:
            raise ValueError(
                f"Threshold {threshold} not in computed thresholds: {self.thresholds}"
            )
        row = df_metrics[df_metrics["threshold"] == threshold].iloc[0]
        return row[self.__metrics_to_calculate()].to_dict()

    def compute_metrics(
        self, threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute and return a dict:
            { "train": {...metrics...}, "test": {...metrics...} }
        """
        tgt = threshold or self.default_threshold
        dfs = {
            split: self._compute_for_df(getattr(self, f"{split}_df"), split)
            for split in ("train", "test")
        }
        return {
            split: self._select_row(df_metrics, tgt)
            for split, df_metrics in dfs.items()
        }

    def plot(self) -> Dict[str, Any]:
        """
        Build and cache:
          - confusion matrices for train & test,
          - reference ROC,
          - optional overlaid train/test ROC area.
        """
        # ensure metrics dataframes exist
        dfs = {
            split: self._compute_for_df(getattr(self, f"{split}_df"), split)
            for split in ("train", "test")
        }

        cm_figs = {
            split: build_confusion_metrics_figure(
                df_metrics=df,
                metrics=self.__metrics_to_calculate(),
                default_idx=self.default_idx,
            )
            for split, df in dfs.items()
        }
        decision_threshold_figs = {
            split: plot_decision_threshold_slider(
                df_metrics=df,
                metrics=self.__metrics_to_calculate(),
                default_threshold=self.default_threshold,
            )
            for split, df in dfs.items()
        }
        # Gain/Lift plot
        gain_lift_figs = {
            split: plot_gain_lift_plotly(
                df_spark=getattr(self, f"{split}_df"),
                label_col=self.label_col,
                bins=10,
                probability_col=self.probability_col,
            )
            for split in ("train", "test")
        }
        report: Dict[str, Any] = {
            "confusion_matrix": cm_figs,
            "roc_reference": plot_reference_rocs(),
            "roc_area": plot_roc_curve_area(dfs["train"], dfs["test"]),
            "pr_reference": plot_reference_pr_curves(),
            "pr_area": plot_pr_curve_area(dfs["train"], dfs["test"]),
            "decision_threshold": decision_threshold_figs,
            "gain_lift": gain_lift_figs,
        }

        self.plots = report
        return report
