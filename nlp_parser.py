import re

FUEL_MAP = {
    "petrol": "Petrol",
    "diesel": "Diesel",
    "cng": "Cng",
    "lpg": "Lpg",
    "electric": "Electric",
    "ev": "Electric",
}

TRANS_MAP = {
    "manual": "Manual",
    "auto": "Automatic",
    "automatic": "Automatic",
}

def parse_budget(text: str):
    """
    Supports:
    - under 1200000
    - under 12 lakh
    - under 1.2 million
    - below 800000
    """
    t = text.lower()

    # million
    m = re.search(r"(under|below|less than)\s*([\d\.]+)\s*(million|m)\b", t)
    if m:
        return int(float(m.group(2)) * 1_000_000)

    # lakh
    m = re.search(r"(under|below|less than)\s*([\d\.]+)\s*(lakh|lakhs)\b", t)
    if m:
        return int(float(m.group(2)) * 100_000)

    # raw number
    m = re.search(r"(under|below|less than)\s*([\d,]+)\b", t)
    if m:
        return int(m.group(2).replace(",", ""))

    return None

def parse_seats(text: str):
    t = text.lower()
    m = re.search(r"(\d+)\s*seats?", t)
    return int(m.group(1)) if m else None

def parse_query(text: str):
    t = text.lower()

    budget_max = parse_budget(t)

    fuel = None
    for k, v in FUEL_MAP.items():
        if k in t:
            fuel = v
            break

    transmission = None
    for k, v in TRANS_MAP.items():
        if k in t:
            transmission = v
            break

    seats_min = parse_seats(t)

    tags = []
    for tag in ["family", "city", "economy", "budget", "luxury", "mileage", "power"]:
        if tag in t:
            tags.append(tag)

    return {
        "budget_max": budget_max,
        "fuel": fuel,
        "transmission": transmission,
        "seats_min": seats_min,
        "tags": tags,
        "raw": text,
    }
    