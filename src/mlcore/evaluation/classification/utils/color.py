# mlcore/evaluation/classification/utils/color.py

from plotly.colors import sample_colorscale

# Configuration for bar color mapping: invert=True for metrics where lower is better
METRICS_CONFIG = {
    "accuracy":           {"invert": False},
    "precision":          {"invert": False},
    "recall":             {"invert": False},
    "f1":                 {"invert": False},
    "specificity":        {"invert": False},
    "npv":                {"invert": False},
    "balanced_accuracy":  {"invert": False},
    "jaccard":            {"invert": False},
    "gmean":              {"invert": False},
    "hamming_loss":       {"invert": True}  # lower is better
}

def metric_color(value: float, invert: bool) -> str:
    """
    Map a metric value to a color on the RdYlGn scale.

    Parameters:
        value (float): Metric value between 0 and 1.
        invert (bool): Whether to invert the scale (for metrics where lower is better).

    Returns:
        str: A Plotly-compatible color string.
    """
    # Invert the value if required
    scaled = 1 - value if invert else value
    # sample_colorscale returns a list; pick the first color
    return sample_colorscale('RdYlGn', [scaled])[0]
