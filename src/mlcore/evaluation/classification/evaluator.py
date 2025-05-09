# mlcore/evaluation/classification/evaluator.py

from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame
from mlcore.evaluation.classification.metrics import compute_threshold_metrics
from mlcore.evaluation.classification.plots import (
    build_confusion_metrics_figure,
    plot_reference_rocs
)

class BinaryClassifierEvaluator:
    """
    Evaluate binary classifiers with thresholded metrics and interactive plots.
    """

    def __init__(
        self,
        spark_df: DataFrame,
        label_col: str = "label",
        probability_col: str = "probability",
        thresholds: Optional[List[float]] = None,
        default_threshold: float = 0.5,
        extra_metrics: bool = False,
    ):
        self.df = spark_df
        self.label_col = label_col
        self.probability_col = probability_col
        self.thresholds = thresholds or [i/20 for i in range(21)]
        self.default_threshold = default_threshold
        self.extra_metrics = extra_metrics
        self.plots: Optional[Dict[str, Any]] = None
        self.threshold_metrics = None
        self.metrics = self.__metrics_to_calculate()
        # find index safely
        try:
            self.default_idx = self.thresholds.index(default_threshold)
        except ValueError:
            self.default_idx = 0

    def __metrics_to_calculate(self) -> List[str]:
        """
        Return a list of metrics to calculate based on the extra_metrics flag.
        """
        base = ["accuracy", "precision", "recall", "f1"]
        extra = [
            "specificity", "npv", "balanced_accuracy",
            "jaccard", "gmean", "hamming_loss"
        ] if self.extra_metrics else []
        return base + extra
    
    def compute_metrics(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute and store per-threshold metrics, returning a dict for one threshold.
        """
        threshold_metrics_df = compute_threshold_metrics(
            df_spark=self.df,
            label_col=self.label_col,
            probability_col=self.probability_col,
            thresholds=self.thresholds,
            extra_metrics=self.extra_metrics
        )
        self.threshold_metrics = threshold_metrics_df
        # Select the requested threshold or default
        target = threshold if threshold is not None else self.default_threshold
        if target not in threshold_metrics_df["threshold"].values:
            raise ValueError(f"Threshold {target} not in computed thresholds: {self.thresholds}")
        row = threshold_metrics_df[threshold_metrics_df["threshold"] == target]
        metrics_dict = row[self.metrics].iloc[0].to_dict()
        return metrics_dict

    def generate_plots(self) -> Dict[str, Any]:
        """
        Generate and return plots: confusion matrix slider and reference ROC curves.
        """
        # Ensure metrics are computed
        if self.threshold_metrics is None:
            self.compute_metrics()

        # Confusion matrix with slider
        cm_fig = build_confusion_metrics_figure(
            df_metrics=self.threshold_metrics,
            metrics=self.metrics,
            default_idx=self.default_idx
        )

        # Reference ROC curves plot
        ref_roc_fig = plot_reference_rocs()

        # Aggregate in report
        report: Dict[str, Any] = {
            "confusion_matrix": cm_fig,
            "roc_reference": ref_roc_fig,
        }
        self.plots = report
        return report
