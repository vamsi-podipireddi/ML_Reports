# mlcore/evaluation/__init__.py

from .classification import BinaryClassifierEvaluator
# in future, add:
# from .classification import MultiClassifierEvaluator
# from .regression import RegressionEvaluator
# from .forecasting import ForecastEvaluator

__all__ = [
    "BinaryClassifierEvaluator",
    # "MultiClassifierEvaluator",
    # "RegressionEvaluator",
    # "ForecastEvaluator",
]
