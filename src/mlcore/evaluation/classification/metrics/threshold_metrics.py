import pandas as pd
from pyspark.sql.functions import col
from typing import List
import math


def compute_threshold_metrics(df_spark, label_col: str, probability_col: str,
                               thresholds: List[float], extra_metrics: bool = False) -> pd.DataFrame:
    """
    Compute confusion matrix counts and classification metrics across given thresholds.

    Parameters:
        df_spark: Input Spark DataFrame.
        label_col (str): Column name for the true label.
        probability_col (str): Column name for prediction probability.
        thresholds (List[float]): List of threshold values to compute metrics.
        extra_metrics (bool): Whether to compute additional metrics.

    Returns:
        pd.DataFrame: DataFrame containing confusion matrix counts and metrics for each threshold.
    """
    # Cast probability and label columns to double for calculation consistency.
    df = df_spark.withColumn("score", col(probability_col).cast("double"))
    df = df.withColumn("label_d", col(label_col).cast("double"))

    rows = []
    for t in thresholds:
        # Compute prediction based on threshold
        pred = df.withColumn("pred_t", (col("score") >= t).cast("double"))
        # Create confusion matrix using pivot operation and fill missing counts with zero
        cm = (pred.groupBy("label_d")
                  .pivot("pred_t", [0.0, 1.0])
                  .count().na.fill(0).orderBy("label_d").collect())
        # Extract TN, FP, FN, TP counts from confusion matrix
        TN, FP = cm[0][1], cm[0][2]
        FN, TP = cm[1][1], cm[1][2]
        total = TP + TN + FP + FN

        # Calculate primary metrics: accuracy, precision, recall, and f1 score.
        acc  = round((TP + TN) / total, 2) if total else 0.0
        prec = round(TP / (TP + FP), 2)    if (TP + FP) else 0.0
        rec  = round(TP / (TP + FN), 2)    if (TP + FN) else 0.0
        f1   = round((2 * prec * rec) / (prec + rec), 2) if (prec + rec) else 0.0

        row = {"threshold": t, "TN": TN, "FP": FP, "FN": FN, "TP": TP,
               "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        # If extra_metrics is True, calculate additional metrics.
        if extra_metrics:
            spec = round(TN / (TN + FP), 2) if (TN + FP) else 0.0
            npv  = round(TN / (TN + FN), 2) if (TN + FN) else 0.0
            bal_acc = round((rec + spec) / 2, 2)
            jaccard = round(TP / (TP + FP + FN), 2) if (TP + FP + FN) else 0.0
            gmean   = round(math.sqrt(rec * spec), 2) if rec * spec >= 0 else 0.0
            h_loss  = round((FP + FN) / total, 2) if total else 0.0
            row.update({"specificity": spec, "npv": npv,
                        "balanced_accuracy": bal_acc,
                        "jaccard": jaccard, "gmean": gmean,
                        "hamming_loss": h_loss})

        rows.append(row)

    return pd.DataFrame(rows)
