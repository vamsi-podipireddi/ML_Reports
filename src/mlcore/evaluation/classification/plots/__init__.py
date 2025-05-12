from .confusion_matrix          import build_confusion_metrics_figure
from .roc_reference             import plot_reference_rocs
from .roc_area                  import plot_roc_curve_area
from .pr_reference              import plot_reference_pr_curves
from .pr_area                   import plot_pr_curve_area
from .decision_threshold        import plot_decision_threshold_slider
from .gain_lift                 import plot_gain_lift_plotly
from .calibration               import plot_calibration_curve_compare  # ← new

__all__ = [
    "build_confusion_metrics_figure",
    "plot_reference_rocs",
    "plot_roc_curve_area",
    "plot_reference_pr_curves",
    "plot_pr_curve_area",
    "plot_decision_threshold_slider",
    "plot_gain_lift_plotly",
    "plot_calibration_curve_compare",  # ← new
]
