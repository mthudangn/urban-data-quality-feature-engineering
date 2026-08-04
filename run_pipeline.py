from __future__ import annotations

from pathlib import Path

import pandas as pd

from delivery_quality import (
    fit_fee_models,
    impute_missing_data,
    plot_fee_model_fit,
    plot_network,
    remove_delivery_fee_outliers,
    repair_dirty_data,
    validate_delivery_data,
)
from property_transformations import (
    plot_feature_correlations,
    plot_skewness_comparison,
    transform_property_features,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    dirty = pd.read_csv(DATA / "delivery_dirty_sample.csv")
    missing = pd.read_csv(DATA / "delivery_missing_sample.csv")
    outlier = pd.read_csv(DATA / "delivery_outlier_sample.csv")
    branches = pd.read_csv(DATA / "branches.csv")
    nodes = pd.read_csv(DATA / "nodes.csv")
    edges = pd.read_csv(DATA / "edges.csv")

    clean_outlier, removed_outliers = remove_delivery_fee_outliers(outlier)
    fee_models = fit_fee_models(clean_outlier)
    clean_dirty = repair_dirty_data(dirty, clean_outlier, fee_models)
    clean_missing = impute_missing_data(missing, branches, nodes, edges, fee_models)

    for output in (clean_outlier, clean_dirty, clean_missing):
        validate_delivery_data(output)

    clean_outlier.to_csv(RESULTS / "delivery_outlier_cleaned.csv", index=False)
    clean_dirty.to_csv(RESULTS / "delivery_dirty_cleaned.csv", index=False)
    clean_missing.to_csv(RESULTS / "delivery_missing_imputed.csv", index=False)
    removed_outliers.to_csv(RESULTS / "delivery_fee_outliers_removed.csv", index=False)

    fee_metrics = pd.DataFrame(
        [
            {"branch": branch, "rows": item.rows, "r2": item.r2, "mae": item.mae}
            for branch, item in sorted(fee_models.items())
        ]
    )
    fee_metrics.to_csv(RESULTS / "delivery_fee_model_metrics.csv", index=False)

    plot_fee_model_fit(clean_outlier, fee_models, FIGURES / "delivery-fee-model-fit.png")
    plot_network(nodes, branches, clean_missing, FIGURES / "delivery-network-reconstruction.png")

    suburb = pd.read_csv(DATA / "melbourne_suburb_sample.csv")
    transformed_suburb, transformation_metrics = transform_property_features(suburb)
    transformed_suburb.to_csv(RESULTS / "melbourne_suburb_transformed.csv", index=False)
    transformation_metrics.to_csv(RESULTS / "property_transformation_metrics.csv", index=False)
    plot_skewness_comparison(suburb, FIGURES / "property-transformation-skewness.png")
    property_correlations = plot_feature_correlations(
        transformed_suburb,
        FIGURES / "property-feature-correlations.png",
    )
    property_correlations.to_csv(RESULTS / "property_feature_correlations.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "dirty_rows_processed", "value": len(dirty)},
            {"metric": "missing_fields_imputed", "value": int(missing.isna().sum().sum())},
            {"metric": "delivery_fee_outliers_removed", "value": len(removed_outliers)},
            {"metric": "suburbs_transformed", "value": len(transformed_suburb)},
        ]
    )
    summary.to_csv(RESULTS / "pipeline_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nDelivery-fee models")
    print(fee_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
