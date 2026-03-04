import os
import joblib
import pandas as pd

from app.nlp_parser import parse_query
from app.utils import explain

DATA_PATH = "vehicles_ai_ready_dataset.csv"
MODEL_PATH = "models/ranker.pkl"

FEATURE_COLS = [
    "budget_max", "q_fuel", "q_trans", "q_seats_min",
    "selling_price", "km_driven", "vehicle_age", "engine_cc",
    "max_power_bhp", "mileage_kmpl", "seats",
    "fuel", "transmission", "brand"
]

# cache in memory (so it doesn't reload every request)
_DF = None
_MODEL = None

def load_assets():
    global _DF, _MODEL
    if _DF is None or _MODEL is None:
        df = pd.read_csv(DATA_PATH).dropna()
        df["selling_price"] = df["selling_price"].astype(float)
        df["km_driven"] = df["km_driven"].astype(float)
        df["seats"] = df["seats"].astype(float)
        _DF = df
        _MODEL = joblib.load(MODEL_PATH)
    return _DF, _MODEL


def recommend(query_text: str, top_n: int = 10):
    df, model = load_assets()
    prefs = parse_query(query_text)

    budget_max = prefs["budget_max"] if prefs["budget_max"] is not None else float(df["selling_price"].max() * 10)

    temp = df.copy()
    temp["budget_max"] = budget_max
    temp["q_fuel"] = prefs["fuel"] if prefs["fuel"] else "Any"
    temp["q_trans"] = prefs["transmission"] if prefs["transmission"] else "Any"
    temp["q_seats_min"] = prefs["seats_min"] if prefs["seats_min"] is not None else 0

    X_live = temp[FEATURE_COLS]
    temp["score"] = model.predict(X_live)

    # small boosts for exact matches
    if prefs["fuel"]:
        temp.loc[temp["fuel"] == prefs["fuel"], "score"] += 0.3
    if prefs["transmission"]:
        temp.loc[temp["transmission"] == prefs["transmission"], "score"] += 0.3
    if prefs["seats_min"] is not None:
        temp.loc[temp["seats"] >= prefs["seats_min"], "score"] += 0.2

    out = temp.sort_values("score", ascending=False).head(int(top_n)).copy()

    results = []
    for _, r in out.iterrows():
        results.append({
            "brand": r["brand"],
            "model": r["model"],
            "price": int(r["selling_price"]),
            "year": int(r["year"]),
            "age": int(r["vehicle_age"]),
            "fuel": r["fuel"],
            "transmission": r["transmission"],
            "km_driven": int(r["km_driven"]),
            "mileage_kmpl": float(r["mileage_kmpl"]),
            "engine_cc": float(r["engine_cc"]),
            "max_power_bhp": float(r["max_power_bhp"]),
            "seats": int(r["seats"]),
            "score": float(r["score"]),
            "why": explain(r, prefs),
        })

    return {"query": prefs, "top_n": int(top_n), "results": results}
if __name__ == "__main__":
    result = recommend(
        "diesel automatic under 800000 family car 5 seats good mileage",
        5
    )
    print(result)