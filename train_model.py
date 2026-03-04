import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from app.utils import pseudo_label

DATA_PATH = "vehicles_ai_ready_dataset.csv"
MODEL_PATH = "models/ranker.pkl"


def build_synthetic_queries():
    return [
        {"budget_max": 300000,  "fuel": "Petrol", "transmission": "Manual",    "seats_min": 5},
        {"budget_max": 500000,  "fuel": "Diesel", "transmission": "Manual",    "seats_min": 5},
        {"budget_max": 700000,  "fuel": "Petrol", "transmission": "Automatic", "seats_min": 5},
        {"budget_max": 900000,  "fuel": "Diesel", "transmission": "Automatic", "seats_min": 5},
        {"budget_max": 1200000, "fuel": "Diesel", "transmission": "Manual",    "seats_min": 7},
        {"budget_max": 800000,  "fuel": "Petrol", "transmission": "Manual",    "seats_min": 5},
        {"budget_max": 600000,  "fuel": None,     "transmission": None,        "seats_min": 5},
    ]


def main():
    df = pd.read_csv(DATA_PATH).dropna()

    # Ensure dtypes
    df["selling_price"] = df["selling_price"].astype(float)
    df["km_driven"] = df["km_driven"].astype(float)
    df["seats"] = df["seats"].astype(float)
    df["vehicle_age"] = df["vehicle_age"].astype(int)

    queries = build_synthetic_queries()

    rows = []
    rng = np.random.default_rng(42)
    sample_size_per_query = 1200  # reduce if slow PC

    for q in queries:
        sample = df.sample(
            n=min(sample_size_per_query, len(df)),
            replace=False,
            random_state=int(rng.integers(1, 1_000_000))
        )

        for _, r in sample.iterrows():
            rec = r.to_dict()
            rec["budget_max"] = q["budget_max"]
            rec["q_fuel"] = q["fuel"] if q["fuel"] is not None else "Any"
            rec["q_trans"] = q["transmission"] if q["transmission"] is not None else "Any"
            rec["q_seats_min"] = q["seats_min"] if q["seats_min"] is not None else 0
            rec["label"] = pseudo_label(r, q)
            rows.append(rec)

    train_df = pd.DataFrame(rows)

    feature_cols = [
        "budget_max", "q_fuel", "q_trans", "q_seats_min",
        "selling_price", "km_driven", "vehicle_age", "engine_cc",
        "max_power_bhp", "mileage_kmpl", "seats",
        "fuel", "transmission", "brand"
    ]

    X = train_df[feature_cols]
    y = train_df["label"]

    cat_cols = ["q_fuel", "q_trans", "fuel", "transmission", "brand"]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # ✅ Works with ALL sklearn versions
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()