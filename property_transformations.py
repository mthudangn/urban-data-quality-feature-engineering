from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox, pearsonr, skew
from sklearn.preprocessing import StandardScaler


def clean_property_data(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["median_income", "median_house_price"]:
        result[column] = (
            result[column]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .astype(float)
        )
    result["aus_born_perc"] = (
        result["aus_born_perc"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )
    numeric = [
        "number_of_houses", "number_of_units", "aus_born_perc",
        "median_income", "median_house_price", "population",
    ]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if (result[numeric] <= 0).any().any():
        raise ValueError("Box-Cox workflow requires strictly positive numeric values")
    return result


def transform_property_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = clean_property_data(df)
    result["std_median_income"] = StandardScaler().fit_transform(result[["median_income"]]).ravel()
    result["boxcox_number_of_units"], lambda_units = boxcox(result["number_of_units"].astype(float))
    result["boxcox_number_of_houses"], lambda_houses = boxcox(result["number_of_houses"].astype(float))
    result["sqrt_population"] = np.sqrt(result["population"].astype(float))
    result["boxcox_median_house_price"], lambda_target = boxcox(result["median_house_price"].astype(float))

    features = [
        "std_median_income",
        "aus_born_perc",
        "boxcox_number_of_units",
        "boxcox_number_of_houses",
        "sqrt_population",
    ]
    correlations = []
    for feature in features:
        r_value, p_value = pearsonr(result[feature], result["boxcox_median_house_price"])
        correlations.append({"feature": feature, "pearson_r": r_value, "p_value": p_value})

    diagnostics = pd.DataFrame(
        [
            {"measure": "boxcox_lambda_number_of_units", "value": lambda_units},
            {"measure": "boxcox_lambda_number_of_houses", "value": lambda_houses},
            {"measure": "boxcox_lambda_median_house_price", "value": lambda_target},
            {"measure": "max_absolute_feature_correlation", "value": result[features].corr().abs().where(~np.eye(len(features), dtype=bool)).max().max()},
        ]
    )
    correlation_df = pd.DataFrame(correlations).sort_values("pearson_r", ascending=False)
    return result, pd.concat([diagnostics, correlation_df.rename(columns={"feature": "measure", "pearson_r": "value"})[["measure", "value"]]], ignore_index=True)


def transformation_skewness(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_property_data(df)
    transformed, _ = transform_property_features(cleaned)
    rows = [
        {"variable": "median_house_price", "before": skew(cleaned["median_house_price"]), "after": skew(transformed["boxcox_median_house_price"])},
        {"variable": "number_of_units", "before": skew(cleaned["number_of_units"]), "after": skew(transformed["boxcox_number_of_units"])},
        {"variable": "number_of_houses", "before": skew(cleaned["number_of_houses"]), "after": skew(transformed["boxcox_number_of_houses"])},
        {"variable": "population", "before": skew(cleaned["population"]), "after": skew(transformed["sqrt_population"])},
    ]
    return pd.DataFrame(rows)


def plot_skewness_comparison(df: pd.DataFrame, output_path: Path) -> None:
    metrics = transformation_skewness(df)
    positions = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(9, 6))
    plt.bar(positions - width / 2, metrics["before"], width=width, label="Before")
    plt.bar(positions + width / 2, metrics["after"], width=width, label="After")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(positions, metrics["variable"], rotation=20, ha="right")
    plt.ylabel("Skewness")
    plt.title("Distribution skewness before and after transformation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_feature_correlations(transformed_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    features = [
        "std_median_income",
        "aus_born_perc",
        "boxcox_number_of_units",
        "boxcox_number_of_houses",
        "sqrt_population",
    ]
    rows = []
    for feature in features:
        r_value, p_value = pearsonr(transformed_df[feature], transformed_df["boxcox_median_house_price"])
        rows.append({"feature": feature, "pearson_r": r_value, "p_value": p_value})
    metrics = pd.DataFrame(rows).sort_values("pearson_r")

    plt.figure(figsize=(9, 6))
    plt.barh(metrics["feature"], metrics["pearson_r"])
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Pearson correlation with transformed house price")
    plt.title("Transformed feature relationships")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return metrics
