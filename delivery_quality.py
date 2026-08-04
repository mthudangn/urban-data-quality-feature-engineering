from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

MEAL_WINDOWS = {
    "Breakfast": (8 * 3600, 12 * 3600),
    "Lunch": (12 * 3600 + 1, 16 * 3600),
    "Dinner": (16 * 3600 + 1, 20 * 3600),
}


@dataclass
class FeeModel:
    model: LinearRegression
    r2: float
    mae: float
    rows: int


def parse_time_seconds(value: str) -> int:
    h, m, s = (int(part) for part in value.split(":"))
    return h * 3600 + m * 60 + s


def expected_order_type(value: str) -> str:
    seconds = parse_time_seconds(value)
    for meal, (start, end) in MEAL_WINDOWS.items():
        if start <= seconds <= end:
            return meal
    raise ValueError(f"Unsupported service time: {value}")


def normalise_date(value: str) -> str:
    formats = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%d-%m",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%Y/%d/%m",
    )
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unparseable date: {value}")


def add_fee_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    result["time_of_day"] = result["time"].map(expected_order_type).map(
        {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
    )
    result["full_delivery_fee"] = np.where(
        result["customerHasloyalty?"].astype(int).eq(1),
        result["delivery_fee"].astype(float) * 2.0,
        result["delivery_fee"].astype(float),
    )
    return result


def fit_fee_models(df: pd.DataFrame) -> Dict[str, FeeModel]:
    featured = add_fee_features(df)
    models: Dict[str, FeeModel] = {}
    for branch, group in featured.groupby("branch_code"):
        X = group[["is_weekend", "time_of_day", "distance_to_customer_KM"]].astype(float)
        y = group["full_delivery_fee"].astype(float)
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)
        models[str(branch)] = FeeModel(
            model=model,
            r2=float(r2_score(y, predictions)),
            mae=float(mean_absolute_error(y, predictions)),
            rows=int(len(group)),
        )
    return models


def remove_delivery_fee_outliers(df: pd.DataFrame, threshold: float = 4.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = df.copy()
    removed_parts: list[pd.DataFrame] = []

    for _ in range(3):
        featured = add_fee_features(current)
        flagged_indices: list[int] = []
        for _, group in featured.groupby("branch_code"):
            X = group[["is_weekend", "time_of_day", "distance_to_customer_KM"]].astype(float)
            y = group["full_delivery_fee"].astype(float)
            model = LinearRegression().fit(X, y)
            residual = y - model.predict(X)
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            if mad <= 1e-12:
                continue
            robust_z = 0.6745 * (residual - median) / mad
            flagged_indices.extend(group.index[np.abs(robust_z) > threshold].tolist())
        if not flagged_indices:
            break
        flagged = current.loc[sorted(set(flagged_indices))].copy()
        removed_parts.append(flagged)
        current = current.drop(index=flagged_indices)

    removed = pd.concat(removed_parts, ignore_index=True) if removed_parts else current.iloc[0:0].copy()
    return current.reset_index(drop=True), removed


def infer_menu_prices(reference_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    prices: Dict[str, Dict[str, float]] = {}
    for meal, group in reference_df.groupby("order_type"):
        items = sorted(
            {
                item
                for raw in group["order_items"]
                for item, _ in ast.literal_eval(str(raw))
            }
        )
        matrix: list[list[float]] = []
        target: list[float] = []
        for row in group.itertuples(index=False):
            quantities = dict(ast.literal_eval(str(row.order_items)))
            matrix.append([float(quantities.get(item, 0)) for item in items])
            target.append(float(row.order_price))
        coefficients, *_ = np.linalg.lstsq(np.asarray(matrix), np.asarray(target), rcond=None)
        prices[str(meal)] = {item: round(float(value), 2) for item, value in zip(items, coefficients)}
    return prices


def compute_order_price(order_items: str, meal: str, menu_prices: Mapping[str, Mapping[str, float]]) -> float:
    total = 0.0
    for item, quantity in ast.literal_eval(str(order_items)):
        total += float(menu_prices[meal][item]) * int(quantity)
    return round(total, 2)


def repair_order_items(order_items: str, order_price: float, meal: str, menu_prices: Mapping[str, Mapping[str, float]]) -> str:
    parsed = list(ast.literal_eval(str(order_items)))
    valid_menu = menu_prices[meal]
    invalid_positions = [i for i, (item, _) in enumerate(parsed) if item not in valid_menu]
    if not invalid_positions:
        return str(parsed)
    if len(invalid_positions) > 1:
        raise ValueError("More than one invalid menu item in a row")

    position = invalid_positions[0]
    _, quantity = parsed[position]
    valid_total = sum(valid_menu[item] * int(qty) for i, (item, qty) in enumerate(parsed) if i != position)
    required_unit_price = (float(order_price) - valid_total) / int(quantity)
    used_items = {item for i, (item, _) in enumerate(parsed) if i != position}
    candidates = [item for item in valid_menu if item not in used_items]
    replacement = min(candidates, key=lambda item: abs(valid_menu[item] - required_unit_price))
    parsed[position] = (replacement, int(quantity))
    return str(parsed)


def repair_coordinates(lat: float, lon: float) -> tuple[float, float]:
    candidates = [
        (lat, lon),
        (-abs(lat), lon),
        (lon, lat),
        (-abs(lon), lat),
        (lon, -abs(lat)),
        (-abs(lon), -abs(lat)),
    ]
    for candidate_lat, candidate_lon in candidates:
        if -38.5 <= candidate_lat <= -37.0 and 144.0 <= candidate_lon <= 146.0:
            return float(candidate_lat), float(candidate_lon)
    raise ValueError(f"Coordinates are outside the expected operating region: {(lat, lon)}")


def predict_full_fee(row: pd.Series, fee_models: Mapping[str, FeeModel]) -> float:
    branch = str(row["branch_code"])
    date = pd.Timestamp(row["date"])
    features = pd.DataFrame(
        [[
            int(date.dayofweek >= 5),
            {"Breakfast": 0, "Lunch": 1, "Dinner": 2}[expected_order_type(str(row["time"]))],
            float(row["distance_to_customer_KM"]),
        ]],
        columns=["is_weekend", "time_of_day", "distance_to_customer_KM"],
    )
    return float(fee_models[branch].model.predict(features)[0])


def repair_dirty_data(
    dirty_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    fee_models: Mapping[str, FeeModel],
) -> pd.DataFrame:
    result = dirty_df.copy()
    menu_prices = infer_menu_prices(reference_df)

    result["date"] = result["date"].map(normalise_date)
    result["branch_code"] = result["branch_code"].astype(str).str.upper()
    result["order_type"] = result["time"].map(expected_order_type)

    for idx, row in result.iterrows():
        result.at[idx, "order_items"] = repair_order_items(
            str(row["order_items"]),
            float(row["order_price"]),
            str(result.at[idx, "order_type"]),
            menu_prices,
        )

    result["order_price"] = result.apply(
        lambda row: compute_order_price(str(row["order_items"]), str(row["order_type"]), menu_prices),
        axis=1,
    )

    repaired_coordinates = result.apply(
        lambda row: repair_coordinates(float(row["customer_lat"]), float(row["customer_lon"])),
        axis=1,
    )
    result["customer_lat"] = [pair[0] for pair in repaired_coordinates]
    result["customer_lon"] = [pair[1] for pair in repaired_coordinates]

    for idx, row in result.iterrows():
        predicted_full = predict_full_fee(row, fee_models)
        actual = float(row["delivery_fee"])
        full_gap = abs(actual - predicted_full)
        discounted_gap = abs(actual - 0.5 * predicted_full)
        result.at[idx, "customerHasloyalty?"] = int(discounted_gap < full_gap)

    return result


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[nx.Graph, cKDTree, np.ndarray]:
    graph = nx.DiGraph()
    for row in edges_df.itertuples(index=False):
        graph.add_edge(int(row.u), int(row.v), weight=float(row.distance_m))
    tree = cKDTree(nodes_df[["lat", "lon"]].to_numpy(dtype=float))
    node_ids = nodes_df["node"].to_numpy(dtype=int)
    return graph, tree, node_ids


def nearest_node(tree: cKDTree, node_ids: np.ndarray, lat: float, lon: float) -> int:
    _, index = tree.query([float(lat), float(lon)])
    return int(node_ids[int(index)])


def impute_missing_data(
    missing_df: pd.DataFrame,
    branches_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    fee_models: Mapping[str, FeeModel],
) -> pd.DataFrame:
    result = missing_df.copy()
    graph, tree, node_ids = build_graph(nodes_df, edges_df)
    branch_nodes = {
        str(row.branch_code): nearest_node(tree, node_ids, float(row.branch_lat), float(row.branch_lon))
        for row in branches_df.itertuples(index=False)
    }
    branch_distance_maps = {
        branch: nx.single_source_dijkstra_path_length(graph, node, weight="weight")
        for branch, node in branch_nodes.items()
    }

    for idx, row in result.iterrows():
        customer_node = nearest_node(tree, node_ids, float(row["customer_lat"]), float(row["customer_lon"]))
        distances = {
            branch: distance_map[customer_node] / 1000.0
            for branch, distance_map in branch_distance_maps.items()
            if customer_node in distance_map
        }
        if not distances:
            raise ValueError(f"No graph path available for customer row {idx}")
        branch_missing = pd.isna(row["branch_code"])
        distance_missing = pd.isna(row["distance_to_customer_KM"])

        if branch_missing and distance_missing:
            selected_branch = min(distances, key=distances.get)
            result.at[idx, "branch_code"] = selected_branch
            result.at[idx, "distance_to_customer_KM"] = round(float(distances[selected_branch]), 3)
        elif branch_missing:
            known_distance = float(row["distance_to_customer_KM"])
            selected_branch = min(distances, key=lambda branch: abs(distances[branch] - known_distance))
            result.at[idx, "branch_code"] = selected_branch
        elif distance_missing:
            branch = str(row["branch_code"])
            result.at[idx, "distance_to_customer_KM"] = round(float(distances[branch]), 3)

    for idx, row in result[result["delivery_fee"].isna()].iterrows():
        full_fee = predict_full_fee(row, fee_models)
        fee = full_fee * (0.5 if int(row["customerHasloyalty?"]) == 1 else 1.0)
        result.at[idx, "delivery_fee"] = fee

    return result


def validate_delivery_data(df: pd.DataFrame) -> None:
    required = {
        "order_id", "date", "time", "order_type", "branch_code", "order_items", "order_price",
        "customer_lat", "customer_lon", "customerHasloyalty?", "distance_to_customer_KM", "delivery_fee",
    }
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Validated output still contains missing values")
    if not set(df["branch_code"]).issubset({"NS", "TP", "BK"}):
        raise ValueError("Unexpected branch code")
    expected = df["time"].map(expected_order_type)
    if not expected.eq(df["order_type"]).all():
        raise ValueError("Order type does not match service time")
    if not df["customer_lat"].between(-38.5, -37.0).all():
        raise ValueError("Latitude outside expected range")
    if not df["customer_lon"].between(144.0, 146.0).all():
        raise ValueError("Longitude outside expected range")


def plot_fee_model_fit(clean_df: pd.DataFrame, fee_models: Mapping[str, FeeModel], output_path: Path) -> None:
    featured = add_fee_features(clean_df)
    actual: list[float] = []
    predicted: list[float] = []
    for _, row in featured.iterrows():
        branch = str(row["branch_code"])
        X = pd.DataFrame(
            [[row["is_weekend"], row["time_of_day"], row["distance_to_customer_KM"]]],
            columns=["is_weekend", "time_of_day", "distance_to_customer_KM"],
            dtype=float,
        )
        actual.append(float(row["full_delivery_fee"]))
        predicted.append(float(fee_models[branch].model.predict(X)[0]))

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.65)
    minimum = min(actual + predicted)
    maximum = max(actual + predicted)
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Actual full delivery fee")
    plt.ylabel("Predicted full delivery fee")
    plt.title("Branch-specific delivery fee model fit")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_network(nodes_df: pd.DataFrame, branches_df: pd.DataFrame, delivery_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    sampled_nodes = nodes_df.sample(n=min(1800, len(nodes_df)), random_state=7)
    plt.scatter(sampled_nodes["lon"], sampled_nodes["lat"], s=4, alpha=0.18)
    plt.scatter(delivery_df["customer_lon"], delivery_df["customer_lat"], s=12, alpha=0.5)
    plt.scatter(branches_df["branch_lon"], branches_df["branch_lat"], s=120, marker="*")
    for row in branches_df.itertuples(index=False):
        plt.annotate(str(row.branch_code), (row.branch_lon, row.branch_lat), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Graph-based branch and customer distance reconstruction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
