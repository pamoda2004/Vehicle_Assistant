import numpy as np

def pseudo_label(vehicle_row, prefs):
    """
    Bootstraps training labels when you don't have real user-click data yet.
    Higher score = better match.
    """
    score = 0.0

    # Budget match
    if prefs.get("budget_max") is not None:
        if vehicle_row["selling_price"] <= prefs["budget_max"]:
            score += 3.0
        else:
            score -= 2.0

        # closer price is better
        ratio = vehicle_row["selling_price"] / max(prefs["budget_max"], 1)
        score += float(np.clip(1.5 - ratio, -1.5, 1.5))

    # Fuel match
    if prefs.get("fuel") is not None:
        score += 2.0 if vehicle_row["fuel"] == prefs["fuel"] else -0.3

    # Transmission match
    if prefs.get("transmission") is not None:
        score += 2.0 if vehicle_row["transmission"] == prefs["transmission"] else -0.3

    # Seats requirement
    if prefs.get("seats_min") is not None:
        score += 1.0 if vehicle_row["seats"] >= prefs["seats_min"] else -0.8

    # Mileage bonus
    score += float(np.clip(vehicle_row["mileage_kmpl"] / 30.0, 0, 1.2))

    # Newer bonus
    age = max(vehicle_row["vehicle_age"], 0)
    score += float(2.0 * (1.0 / (1.0 + age)))

    # Power bonus
    score += float(np.clip(vehicle_row["max_power_bhp"] / 150.0, 0, 1.0))

    return score


def explain(vehicle_row, prefs):
    reasons = []

    if prefs.get("budget_max") is not None:
        if vehicle_row["selling_price"] <= prefs["budget_max"]:
            reasons.append("Within your budget")
        else:
            reasons.append("Above budget but good overall match")

    if prefs.get("fuel") and vehicle_row["fuel"] == prefs["fuel"]:
        reasons.append(f"Fuel match: {vehicle_row['fuel']}")

    if prefs.get("transmission") and vehicle_row["transmission"] == prefs["transmission"]:
        reasons.append(f"Transmission match: {vehicle_row['transmission']}")

    if prefs.get("seats_min") is not None and vehicle_row["seats"] >= prefs["seats_min"]:
        reasons.append(f"Seats: {int(vehicle_row['seats'])} meets requirement")

    if vehicle_row["vehicle_age"] <= 5:
        reasons.append("Relatively newer vehicle")

    if vehicle_row["mileage_kmpl"] >= 20:
        reasons.append("Good mileage")

    return reasons[:4]