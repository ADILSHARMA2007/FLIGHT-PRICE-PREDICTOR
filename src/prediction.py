"""
Flight Price Predictor - ML Powered (v2.3)
=============================================
Supports BOTH US domestic and Indian domestic flights.

Uses trained XGBoost ML models for BOTH markets:
  - US routes: ML pipeline trained on US flight dataset
  - Indian routes: ML pipeline trained on 300K Indian flight records

Both markets show real dataset comparison statistics alongside
ML predictions for transparency and validation.

Supported markets:
  US Domestic  — 50+ airports, USD pricing
  India Domestic — 6 metro cities, INR pricing (Economy + Business)
"""

import os
import pandas as pd
import numpy as np
import joblib
import random
import math

# Project root = parent of src/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load US model artifacts
BASE_DIR = os.path.join(_PROJECT_ROOT, "models")

pipeline = joblib.load(os.path.join(BASE_DIR, "flight_pipeline_v2.pkl"))
route_encoding = joblib.load(os.path.join(BASE_DIR, "route_encoding_v2.pkl"))
global_mean = joblib.load(os.path.join(BASE_DIR, "global_mean_v2.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns_v2.pkl"))

# Load Indian model artifacts
in_best_model = joblib.load(os.path.join(BASE_DIR, "indian_flight_best_pipeline.joblib"))

INDIAN_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "clean_dataset.csv")
in_df_full = pd.read_csv(INDIAN_DATA_PATH).copy()
for _c in ["airline", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class"]:
    in_df_full[_c] = in_df_full[_c].astype(str).str.strip().str.lower()
in_df_full["route"] = in_df_full["source_city"] + "_" + in_df_full["destination_city"]

# Feature maps aligned with BASE/indian/feature_engineering.ipynb
IN_TIME_ORDER = {
    "early_morning": 0,
    "morning": 1,
    "afternoon": 2,
    "evening": 3,
    "night": 4,
    "late_night": 5,
}
IN_STOPS_ORDER = {"zero": 0, "one": 1, "two_or_more": 2}
IN_CLASS_ORDER = {"economy": 0, "business": 1}
IN_ADVANCE_BINS = [-1, 2, 7, 15, 30, 60, 120, 365]
IN_ADVANCE_LABELS = ["very_last_min", "last_min", "short", "medium", "long", "very_long", "ultra_long"]

in_global_mean = float(in_df_full["price"].mean())
IN_FREQ_COLS = ["airline", "source_city", "destination_city", "route"]
in_freq_maps = {
    col: in_df_full[col].value_counts(normalize=True).to_dict()
    for col in IN_FREQ_COLS
}

in_route_stats = in_df_full.groupby("route")["price"].agg(["count", "mean"])
IN_ROUTE_SMOOTHING = 30
in_route_stats["smooth"] = (
    in_route_stats["count"] * in_route_stats["mean"] + IN_ROUTE_SMOOTHING * in_global_mean
) / (in_route_stats["count"] + IN_ROUTE_SMOOTHING)
in_route_te_map = in_route_stats["smooth"].to_dict()

# Indian city name <-> IATA code mapping (dataset uses lowercase city names)
INDIAN_CITY_TO_IATA = {
    "delhi": "DEL", "mumbai": "BOM", "bangalore": "BLR",
    "kolkata": "CCU", "hyderabad": "HYD", "chennai": "MAA",
}
INDIAN_IATA_TO_CITY = {v: k for k, v in INDIAN_CITY_TO_IATA.items()}


def indian_time_bin(hour_24):
    if 4 <= hour_24 < 8:
        return "early_morning"
    if 8 <= hour_24 < 12:
        return "morning"
    if 12 <= hour_24 < 16:
        return "afternoon"
    if 16 <= hour_24 < 20:
        return "evening"
    if 20 <= hour_24 < 24:
        return "night"
    return "late_night"


def build_indian_model_input(airline, source_city, departure_time, stops, arrival_time,
                             destination_city, flight_class, duration, days_left):
    route = f"{source_city}_{destination_city}"
    stops_ord = IN_STOPS_ORDER.get(stops, 3)
    class_ord = IN_CLASS_ORDER.get(flight_class, -1)
    dep_ord = IN_TIME_ORDER.get(departure_time, -1)
    arr_ord = IN_TIME_ORDER.get(arrival_time, -1)

    row = {
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": flight_class,
        "duration": float(duration),
        "days_left": int(days_left),
        "route": route,
        "is_same_city": int(source_city == destination_city),
        "departure_time_ord": int(dep_ord),
        "arrival_time_ord": int(arr_ord),
        "stops_ord": int(stops_ord),
        "class_ord": int(class_ord),
        "dep_arr_diff_ord": int((arr_ord - dep_ord) % 6),
        "duration_per_stop": float(duration) / (int(stops_ord) + 1),
        "duration_x_stops": float(duration) * (int(stops_ord) + 1),
        "advance_booking_bucket": str(
            pd.cut([days_left], bins=IN_ADVANCE_BINS, labels=IN_ADVANCE_LABELS)[0]
        ),
        "log_duration": float(np.log1p(duration)),
        "log_days_left": float(np.log1p(days_left)),
        "airline_freq": float(in_freq_maps["airline"].get(airline, 0.0)),
        "source_city_freq": float(in_freq_maps["source_city"].get(source_city, 0.0)),
        "destination_city_freq": float(in_freq_maps["destination_city"].get(destination_city, 0.0)),
        "route_freq": float(in_freq_maps["route"].get(route, 0.0)),
        "route_te": float(in_route_te_map.get(route, in_global_mean)),
    }

    model_cols = list(in_best_model.feature_names_in_)
    return pd.DataFrame([row], columns=model_cols)

# US Airline Tiers
BUDGET_AIRLINES_US = {
    "Spirit Air Lines", "Frontier Airlines Inc.", "Allegiant Air"
}
MID_TIER_AIRLINES_US = {
    "Southwest Airlines Co.", "JetBlue Airways", "Virgin America",
    "Alaska Airlines Inc.", "Hawaiian Airlines Inc."
}
PREMIUM_AIRLINES_US = {
    "Delta Air Lines Inc.", "United Air Lines Inc.",
    "American Airlines Inc.", "US Airways Inc."
}
REGIONAL_AIRLINES_US = {
    "Atlantic Southeast Airlines", "Skywest Airlines Inc.",
    "American Eagle Airlines Inc."
}


# Airport coordinates for distance estimation (US & India)
AIRPORT_COORDS = {
    # US Airports
    "ATL": (33.6407, -84.4277), "LAX": (33.9425, -118.4081), "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403), "DEN": (39.8561, -104.6737), "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790), "SEA": (47.4502, -122.3088), "LAS": (36.0840, -115.1537),
    "MCO": (28.4312, -81.3081), "EWR": (40.6895, -74.1745), "MSP": (44.8848, -93.2223),
    "BOS": (42.3656, -71.0096), "DTW": (42.2162, -83.3554), "PHL": (39.8744, -75.2424),
    "LGA": (40.7769, -73.8740), "FLL": (26.0742, -80.1506), "BWI": (39.1774, -76.6684),
    "IAH": (29.9902, -95.3368), "SLC": (40.7884, -111.9778), "DCA": (38.8512, -77.0402),
    "SAN": (32.7338, -117.1933), "MDW": (41.7868, -87.7522), "TPA": (27.9756, -82.5333),
    "PDX": (45.5898, -122.5951), "HNL": (21.3187, -157.9224), "STL": (38.7487, -90.3700),
    "MIA": (25.7959, -80.2870), "BNA": (36.1263, -86.6774), "AUS": (30.1975, -97.6664),
    "HOU": (29.6454, -95.2789), "OAK": (37.7213, -122.2208), "MSY": (29.9934, -90.2580),
    "SJC": (37.3626, -121.9290), "RDU": (35.8801, -78.7880), "CLE": (41.4117, -81.8498),
    "SAT": (29.5337, -98.4698), "SMF": (38.6954, -121.5908), "PIT": (40.4957, -80.2413),
    "IND": (39.7173, -86.2944), "CVG": (39.0488, -84.6678), "CMH": (39.9980, -82.8919),
    "MCI": (39.2976, -94.7139), "JAX": (30.4941, -81.6879), "SNA": (33.6757, -117.8683),
    "ABQ": (35.0402, -106.6090), "ANC": (61.1744, -149.9964), "PHX": (33.4373, -112.0078),
    "MKE": (42.9472, -87.8966), "RIC": (37.5052, -77.3197), "RSW": (26.5362, -81.7552),
    "BUF": (42.9405, -78.7324), "PVD": (41.7249, -71.4283), "OMA": (41.3032, -95.8941),
    "CHS": (32.8986, -80.0405), "ORF": (36.8946, -76.2012), "CLT": (35.2140, -80.9431),
    "DAL": (32.8471, -96.8518), "ONT": (34.0560, -117.6012), "BUR": (34.2005, -118.3585),

    # India — Metro / Tier-1
    "DEL": (28.5562, 77.1000),   # Delhi (Indira Gandhi)
    "BOM": (19.0896, 72.8656),   # Mumbai (Chhatrapati Shivaji)
    "BLR": (13.1986, 77.7066),   # Bengaluru (Kempegowda)
    "MAA": (12.9941, 80.1709),   # Chennai
    "CCU": (22.6547, 88.4467),   # Kolkata (Netaji Subhash Chandra Bose)
    "HYD": (17.2403, 78.4294),   # Hyderabad (Rajiv Gandhi)
    "COK": (10.1520, 76.4019),   # Kochi
    "GOI": (15.3809, 73.8314),   # Goa (Dabolim)
    "GOX": (15.7383, 73.8319),   # Goa (Manohar/Mopa)
    "PNQ": (18.5822, 73.9197),   # Pune
    "AMD": (23.0772, 72.6347),   # Ahmedabad
    "JAI": (26.8242, 75.8122),   # Jaipur
    "LKO": (26.7606, 80.8893),   # Lucknow
    "GAU": (26.1061, 91.5859),   # Guwahati
    "TRV": (8.4821, 76.9201),    # Thiruvananthapuram (Trivandrum)

    # India — Tier-2
    "SXR": (33.9871, 74.7742),   # Srinagar
    "IXC": (30.6735, 76.7885),   # Chandigarh
    "PAT": (25.5913, 85.0880),   # Patna
    "BBI": (20.2444, 85.8178),   # Bhubaneswar
    "IXR": (23.3143, 85.3217),   # Ranchi
    "NAG": (21.0922, 79.0472),   # Nagpur
    "VNS": (25.4524, 82.8593),   # Varanasi
    "IXB": (26.6812, 88.3286),   # Bagdogra (Siliguri)
    "IDR": (22.7218, 75.8011),   # Indore
    "VTZ": (17.7232, 83.2245),   # Visakhapatnam (Vizag)
    "RPR": (21.1804, 81.7388),   # Raipur
    "DED": (30.1897, 78.1803),   # Dehradun (Jolly Grant)
    "IXA": (23.8870, 91.2404),   # Agartala
    "IMF": (24.7600, 93.8967),   # Imphal
    "DIB": (27.4839, 95.0169),   # Dibrugarh
    "CJB": (11.0300, 77.0434),   # Coimbatore
    "IXM": (9.8345, 78.0934),    # Madurai
    "TRZ": (10.7654, 78.7098),   # Tiruchirappalli (Trichy)
    "CCJ": (11.1368, 75.9553),   # Kozhikode (Calicut)
    "IXE": (12.9613, 74.8901),   # Mangalore
    "CNN": (11.9186, 75.5471),   # Kannur
    "BHO": (23.2875, 77.3374),   # Bhopal
    "GWL": (26.2933, 78.2278),   # Gwalior
    "UDR": (24.6177, 73.8961),   # Udaipur
    "JDH": (26.2511, 73.0489),   # Jodhpur
    "BDQ": (22.3362, 73.2264),   # Vadodara
    "RAJ": (22.3092, 70.7795),   # Rajkot
    "STV": (21.1141, 72.7418),   # Surat
    "IXJ": (32.6891, 74.8374),   # Jammu
    "IXL": (34.1359, 77.5465),   # Leh
    "KUU": (31.8767, 77.1544),   # Kullu (Bhuntar)
    "DHM": (32.1651, 76.2634),   # Dharamsala (Kangra)

    # India — Tier-3 / Regional
    "IXZ": (11.6412, 92.7297),   # Port Blair (Andaman)
    "SHL": (25.7036, 91.9787),   # Shillong (Umroi)
    "DMU": (25.8839, 93.7711),   # Dimapur
    "AJL": (23.7462, 92.8027),   # Aizawl (Lengpui)
    "AGR": (27.1557, 77.9609),   # Agra (Kheria)
    "MYQ": (12.2300, 76.6557),   # Mysore
    "IXG": (15.8593, 74.6183),   # Belgaum (Belagavi)
    "HBX": (15.3617, 75.0849),   # Hubli
    "VGA": (16.5304, 80.7968),   # Vijayawada (Gannavaram)
    "TIR": (13.6325, 79.5433),   # Tirupati
    "RJA": (17.1104, 81.8182),   # Rajahmundry
    "IXD": (25.4401, 81.7340),   # Prayagraj (Allahabad)
    "JRH": (26.7315, 94.1753),   # Jorhat
    "HJR": (24.8172, 79.9186),   # Khajuraho
    "BHJ": (23.2878, 69.6702),   # Bhuj
    "PBD": (21.6487, 69.6573),   # Porbandar
    "KNU": (26.4041, 80.4101),   # Kanpur (Chakeri)
    "JSA": (26.8887, 70.8650),   # Jaisalmer
    "PGH": (21.1788, 81.8843),   # Pantnagar – approx
    "IXW": (22.8132, 86.1688),   # Jamshedpur (Sonari)
    "SAG": (21.1141, 72.7418),   # Shirdi
    "ISK": (20.1191, 73.9130),   # Nashik (Ozar)
    "KLR": (24.0000, 79.9200),   # Kolhapur – approx
    "TCR": (8.7247, 76.9200),    # Tuticorin
    "PUT": (11.1200, 76.0300),   # Puducherry – approx (Sri Sathya Sai)
    "GOP": (26.7397, 83.4496),   # Gorakhpur
    "SLV": (24.1503, 88.2459),   # Malda – approx (Silchar actually)
    "IXS": (24.9130, 92.9787),   # Silchar
    "TEZ": (26.7091, 92.7847),   # Tezpur
    "LDA": (26.7606, 81.0044),   # Malda – approx
    "GAY": (24.7443, 84.9512),   # Gaya
    "JLR": (23.1778, 80.0520),   # Jabalpur
    "DHH": (23.8328, 72.1897),   # Dhanbad – approx
    "BEP": (15.8585, 74.6182),   # Bellary – approx
}

# Indian airport IATA codes (for auto-detection)
INDIAN_AIRPORTS = {
    "DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "GOI", "GOX", "PNQ",
    "AMD", "JAI", "LKO", "GAU", "TRV", "SXR", "IXC", "PAT", "BBI", "IXR",
    "NAG", "VNS", "IXB", "IDR", "VTZ", "RPR", "DED", "IXA", "IMF", "DIB",
    "CJB", "IXM", "TRZ", "CCJ", "IXE", "CNN", "BHO", "GWL", "UDR", "JDH",
    "BDQ", "RAJ", "STV", "IXJ", "IXL", "KUU", "DHM", "IXZ", "SHL", "DMU",
    "AJL", "AGR", "MYQ", "IXG", "HBX", "VGA", "TIR", "RJA", "IXD", "JRH",
    "HJR", "BHJ", "PBD", "KNU", "JSA", "PGH", "IXW", "SAG", "ISK", "KLR",
    "TCR", "PUT", "GOP", "IXS", "TEZ", "GAY", "JLR",
}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in kilometers."""
    return haversine_miles(lat1, lon1, lat2, lon2) * 1.60934


def estimate_distance(origin, destination, unit="miles"):
    """Estimate flight distance using airport coordinates."""
    if origin in AIRPORT_COORDS and destination in AIRPORT_COORDS:
        lat1, lon1 = AIRPORT_COORDS[origin]
        lat2, lon2 = AIRPORT_COORDS[destination]
        if unit == "km":
            return round(haversine_km(lat1, lon1, lat2, lon2))
        return round(haversine_miles(lat1, lon1, lat2, lon2))
    return None


def is_indian_route(origin, destination):
    """Check if both airports are Indian."""
    return origin in INDIAN_AIRPORTS and destination in INDIAN_AIRPORTS




# Airline Tier Classification

def get_airline_tier(airline_name, market="US"):
    """Classify airline into pricing tier."""
    name_lower = airline_name.lower()

    if market == "IN":
        # Indian airlines
        if any(kw in name_lower for kw in ["indigo", "6e"]):
            return "budget"
        elif any(kw in name_lower for kw in ["spicejet", "spice", "sg"]):
            return "budget"
        elif any(kw in name_lower for kw in ["akasa", "qp"]):
            return "budget"
        elif any(kw in name_lower for kw in ["go first", "g8"]):
            return "budget"
        elif any(kw in name_lower for kw in ["air india express", "ix"]):
            return "budget"
        elif any(kw in name_lower for kw in ["air india", "vistara", "ai", "uk"]):
            return "premium"
        elif any(kw in name_lower for kw in ["alliance", "star air", "flybig", "trujet"]):
            return "regional"
        else:
            return "budget"  # Most Indian flyers use LCCs
    else:
        # US airlines
        if airline_name in BUDGET_AIRLINES_US:
            return "budget"
        elif airline_name in MID_TIER_AIRLINES_US:
            return "mid"
        elif airline_name in PREMIUM_AIRLINES_US:
            return "premium"
        elif airline_name in REGIONAL_AIRLINES_US:
            return "regional"
        # Fuzzy matching
        if any(kw in name_lower for kw in ["spirit", "frontier", "allegiant"]):
            return "budget"
        elif any(kw in name_lower for kw in ["southwest", "jetblue", "virgin", "alaska", "hawaiian"]):
            return "mid"
        elif any(kw in name_lower for kw in ["delta", "united", "american", "us airways"]):
            return "premium"
        return "mid"





# User Input Handling
print("\n  Flight Price Predictor v2.3 (US + India) - ML Powered")
print("=" * 60)
print("  Supports: US domestic  |  India domestic")
print("=" * 60)

origin = input("\nEnter origin airport code (e.g., DEL, LAX): ").strip().upper()
destination = input("Enter destination airport code (e.g., BOM, JFK): ").strip().upper()
date = input("Enter travel date (YYYY-MM-DD): ").strip()
departure_time = input("Enter departure time (HH:MM): ").strip()

# Detect market
india_mode = is_indian_route(origin, destination)

if india_mode:
    print("\n   🇮🇳 Indian route detected — prices will be in ₹ (INR)")
    airline_prompt = "Enter airline (e.g., IndiGo, SpiceJet, Air India, Vistara, Akasa): "
else:
    airline_prompt = "Enter airline (e.g., Delta, Spirit, Southwest, United): "

airline = input(airline_prompt).strip()
days_ahead_input = input("How many days before departure are you booking? (e.g., 14): ").strip()

# India-specific booking attributes (improves ML feature alignment)
flight_class = "Economy"
stops_label = "one"
if india_mode:
    class_input = input("Enter class (Economy/Business) [default Economy]: ").strip().lower()
    if class_input in {"business", "b", "biz"}:
        flight_class = "Business"

    stops_input = input("Enter stops (zero/one/two_or_more) [default one]: ").strip().lower()
    STOPS_ALIASES = {
        "": "one",
        "0": "zero", "zero": "zero", "nonstop": "zero", "non-stop": "zero", "direct": "zero",
        "1": "one", "one": "one", "one-stop": "one", "1-stop": "one",
        "2": "two_or_more", "2+": "two_or_more", "two": "two_or_more", "two_or_more": "two_or_more",
        "multi": "two_or_more", "multiple": "two_or_more",
    }
    if stops_input not in STOPS_ALIASES:
        print("   [WARN] Unknown stops value, defaulting to 'one'")
    stops_label = STOPS_ALIASES.get(stops_input, "one")

# Normalize airline name (aliases)
AIRLINE_ALIASES = {
    # US
    "delta": "Delta Air Lines Inc.", "united": "United Air Lines Inc.",
    "american": "American Airlines Inc.", "southwest": "Southwest Airlines Co.",
    "spirit": "Spirit Air Lines", "frontier": "Frontier Airlines Inc.",
    "jetblue": "JetBlue Airways", "alaska": "Alaska Airlines Inc.",
    "hawaiian": "Hawaiian Airlines Inc.", "virgin": "Virgin America",
    "us airways": "US Airways Inc.",
    # India (must match the dataset airline names exactly)
    "indigo": "Indigo", "6e": "Indigo",
    "spicejet": "SpiceJet", "spice": "SpiceJet", "sg": "SpiceJet",
    "air india": "Air India", "ai": "Air India",
    "vistara": "Vistara", "uk": "Vistara",
    "akasa": "Akasa Air", "akasa air": "Akasa Air", "qp": "Akasa Air",
    "go first": "GO FIRST", "g8": "GO FIRST",
    "air india express": "Air India", "ix": "Air India",
    "airasia": "AirAsia", "i5": "AirAsia",
    "alliance air": "Air India", "star air": "SpiceJet",
}
airline_normalized = AIRLINE_ALIASES.get(airline.lower(), airline)

# Parse Inputs
date_dt = pd.to_datetime(date)
month = date_dt.month
day = date_dt.day
day_of_week = date_dt.weekday()

time_parts = departure_time.split(":")
dep_hour = int(time_parts[0])
dep_min = int(time_parts[1]) if len(time_parts) > 1 else 0

try:
    days_until_departure = int(days_ahead_input)
except ValueError:
    days_until_departure = 21

is_weekend = 1 if day_of_week >= 5 else 0
is_holiday_season = 1 if month in [11, 12] else 0

# Distance Calculation
route = f"{origin}_{destination}"

if india_mode:
    # Indian routes: use geo-distance in km
    dist_km = estimate_distance(origin, destination, unit="km")
    if dist_km:
        distance_km = dist_km
        distance_miles = round(dist_km / 1.60934)
    else:
        distance_km = 1000
        distance_miles = 621
        print(f"   [WARN] Airport not found, using default distance: {distance_km} km")
    scheduled_time = (distance_miles / 480) * 60 + 25  # Indian domestic ~480mph avg

    # Use preloaded Indian dataset for comparison
    print("\n   Loading Indian flight dataset...")
    df_india = in_df_full

    # Map IATA codes to city names for dataset lookup
    origin_city = INDIAN_IATA_TO_CITY.get(origin, None)
    dest_city = INDIAN_IATA_TO_CITY.get(destination, None)

    # Dataset variables
    in_ds_route_stats = None
    in_ds_airline_stats = None
    in_ds_similar_stats = None
    in_route_data = pd.DataFrame()
    airline_in_india_dataset = False

    if origin_city and dest_city:
        # Filter to selected class to keep model output and comparison aligned
        model_class = flight_class.lower()
        df_india_econ = df_india[df_india["class"] == model_class].copy()
        df_india_econ["route"] = df_india_econ["source_city"] + "_" + df_india_econ["destination_city"]
        india_route_key = f"{origin_city}_{dest_city}"
        in_route_data = df_india_econ[df_india_econ["route"] == india_route_key]

        dataset_airlines_india = set(df_india_econ["airline"].unique())
        airline_in_india_dataset = airline_normalized.lower() in dataset_airlines_india

        if not airline_in_india_dataset:
            print(f"   [WARN] Airline '{airline_normalized}' not in Indian dataset.")
            print(f"   Available: {', '.join(sorted(dataset_airlines_india))}")

        if len(in_route_data) > 0:
            in_ds_route_stats = {
                "mean": round(in_route_data["price"].mean()),
                "min": round(in_route_data["price"].min()),
                "max": round(in_route_data["price"].max()),
                "median": round(in_route_data["price"].median()),
                "count": len(in_route_data),
            }

            if airline_in_india_dataset:
                route_airline = in_route_data[in_route_data["airline"] == airline_normalized.lower()]
                if len(route_airline) > 0:
                    in_ds_airline_stats = {
                        "mean": round(route_airline["price"].mean()),
                        "min": round(route_airline["price"].min()),
                        "max": round(route_airline["price"].max()),
                        "median": round(route_airline["price"].median()),
                        "count": len(route_airline),
                    }

            # Similar conditions: same days_left bucket
            dl_bucket_min = max(1, days_until_departure - 5)
            dl_bucket_max = days_until_departure + 5
            similar = in_route_data[
                (in_route_data["days_left"] >= dl_bucket_min) &
                (in_route_data["days_left"] <= dl_bucket_max)
            ]
            if len(similar) >= 5:
                in_ds_similar_stats = {
                    "mean": round(similar["price"].mean()),
                    "min": round(similar["price"].min()),
                    "max": round(similar["price"].max()),
                    "count": len(similar),
                }
    else:
        print(f"   [INFO] Cities {origin}/{destination} not in Indian dataset (6 metros only).")
        print(f"   Supported: DEL, BOM, BLR, CCU, HYD, MAA")
        print(f"   Using ML model with calibration fallback.")
else:
    # US routes: Load ALL data from the dataset
    print("\n   📂 Loading dataset...")
    df_full = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "processed", "clean_dataset1.csv"))

    # --- Get valid airports & airlines from dataset ---
    dataset_airports = set(df_full["ORIGIN_AIRPORT"].unique()) | set(df_full["DESTINATION_AIRPORT"].unique())
    dataset_airlines = set(df_full["AIRLINE"].unique())

    # --- Validate origin & destination against dataset ---
    if origin not in dataset_airports:
        print(f"\n   ❌ Airport '{origin}' not found in dataset.")
        print(f"   Available airports ({len(dataset_airports)}): {', '.join(sorted(dataset_airports))}")
        exit()
    if destination not in dataset_airports:
        print(f"\n   ❌ Airport '{destination}' not found in dataset.")
        print(f"   Available airports ({len(dataset_airports)}): {', '.join(sorted(dataset_airports))}")
        exit()

    # --- Check if airline exists in dataset ---
    airline_in_dataset = airline_normalized in dataset_airlines
    if not airline_in_dataset:
        print(f"\n   ⚠️  Airline '{airline_normalized}' not in dataset.")
        print(f"   Available airlines: {', '.join(sorted(dataset_airlines))}")
        print(f"   → Will still predict using ML model, but no dataset comparison available for this airline.")

    # --- Get distance & scheduled_time from dataset ---
    df_full["ROUTE"] = df_full["ORIGIN_AIRPORT"] + "_" + df_full["DESTINATION_AIRPORT"]
    route_data = df_full[df_full["ROUTE"] == route]

    if len(route_data) > 0:
        distance_miles = route_data["DISTANCE"].iloc[0]
        scheduled_time = route_data["SCHEDULED_TIME"].median()
    else:
        # Route not in dataset — try geo-fallback
        geo_dist = estimate_distance(origin, destination)
        if geo_dist:
            distance_miles = geo_dist
            scheduled_time = (distance_miles / 500) * 60 + 30
        else:
            distance_miles = 1000
            scheduled_time = 150.0
        print(f"   ℹ️  Route {route} not in dataset, estimated distance: {int(distance_miles)} mi")

    # Dataset fare stats (actual prices from your data)
    ds_route_stats = None
    ds_route_airline_stats = None
    ds_similar_stats = None

    # Stats for this exact route (all airlines)
    if len(route_data) > 0:
        ds_route_stats = {
            "mean": round(route_data["FARE"].mean(), 2),
            "min": round(route_data["FARE"].min(), 2),
            "max": round(route_data["FARE"].max(), 2),
            "median": round(route_data["FARE"].median(), 2),
            "count": len(route_data),
        }

        # Stats for this route + specific airline
        if airline_in_dataset:
            route_airline_data = route_data[route_data["AIRLINE"] == airline_normalized]
            if len(route_airline_data) > 0:
                ds_route_airline_stats = {
                    "mean": round(route_airline_data["FARE"].mean(), 2),
                    "min": round(route_airline_data["FARE"].min(), 2),
                    "max": round(route_airline_data["FARE"].max(), 2),
                    "median": round(route_airline_data["FARE"].median(), 2),
                    "count": len(route_airline_data),
                }

        # Stats for similar conditions (same route + same month + same day type)
        similar = route_data[
            (route_data["MONTH"] == month) &
            (route_data["IS_WEEKEND"] == is_weekend)
        ]
        if len(similar) >= 5:
            ds_similar_stats = {
                "mean": round(similar["FARE"].mean(), 2),
                "min": round(similar["FARE"].min(), 2),
                "max": round(similar["FARE"].max(), 2),
                "count": len(similar),
            }

# Predict Fare
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

if india_mode:
    # INDIA: ML model prediction (trained on 300K Indian flights)
    origin_city = INDIAN_IATA_TO_CITY.get(origin, None)
    dest_city = INDIAN_IATA_TO_CITY.get(destination, None)

    # Estimate duration using realistic block speed + layover time by stops
    layover_hours = {"zero": 0.0, "one": 2.0, "two_or_more": 4.0}
    base_block_hours = max(1.0, (distance_km / 560.0) + 0.25)
    est_duration_hours = base_block_hours + layover_hours.get(stops_label, 2.0)

    # Departure/arrival bins aligned with feature_engineering.ipynb
    in_dep_time_bin = indian_time_bin(dep_hour)
    arr_hour_est = int((dep_hour + est_duration_hours) % 24)
    in_arr_time_bin = indian_time_bin(arr_hour_est)

    # City mapping is required because the Indian training data is city-based.
    can_use_ml = origin_city is not None and dest_city is not None

    if can_use_ml:
        df_input = build_indian_model_input(
            airline=airline_normalized.lower(),
            source_city=origin_city,
            departure_time=in_dep_time_bin,
            stops=stops_label,
            arrival_time=in_arr_time_bin,
            destination_city=dest_city,
            flight_class=flight_class.lower(),
            duration=round(est_duration_hours, 2),
            days_left=days_until_departure,
        )

        raw_prediction = float(in_best_model.predict(df_input)[0])
        base_model_label = "Indian Best Pipeline"


        fare_model = max(raw_prediction, 1100.0)

        # Anchor prediction to historical route/airline statistics when available.
        anchor_price = None
        if in_ds_airline_stats:
            anchor_price = in_ds_airline_stats["mean"]
        elif in_ds_route_stats:
            anchor_price = in_ds_route_stats["mean"]
        if anchor_price is not None and in_ds_similar_stats:
            anchor_price = 0.65 * anchor_price + 0.35 * in_ds_similar_stats["mean"]

        adjusted_prediction = fare_model
        if anchor_price is not None:
            # Business fares are more volatile and often under-predicted by pure model output.
            if flight_class == "Business":
                model_weight = 0.55
                anchor_weight = 0.45
            else:
                model_weight = 0.45
                anchor_weight = 0.55
            adjusted_prediction = model_weight * fare_model + anchor_weight * anchor_price

            ratio = anchor_price / max(fare_model, 1)
            if ratio > 1.5:
                adjusted_prediction *= 1.10
            elif ratio > 1.25:
                adjusted_prediction *= 1.05

        if in_ds_airline_stats:
            floor = in_ds_airline_stats["min"] * 0.95
            ceiling = in_ds_airline_stats["max"] * 1.05
        elif in_ds_route_stats:
            floor = in_ds_route_stats["min"] * 0.95
            ceiling = in_ds_route_stats["max"] * 1.05
        else:
            floor = 1100
            ceiling = 50000

        if anchor_price:
            min_allowed = anchor_price * 0.75
        else:
            min_allowed = floor

        fare = round(min(max(adjusted_prediction, min_allowed), ceiling))

        # Range from dataset if available, else estimate
        if in_ds_airline_stats:
            low = in_ds_airline_stats["min"]
            high = in_ds_airline_stats["max"]
        elif in_ds_route_stats:
            low = in_ds_route_stats["min"]
            high = in_ds_route_stats["max"]
        else:
            low = round(fare * 0.78)
            high = round(fare * 1.25)

        if anchor_price is not None:
            prediction_source = f"ML Model ({base_model_label}) + historical route correction"
        else:
            prediction_source = f"ML Model ({base_model_label})"
    else:
        # Fallback: simple distance-based estimate for cities not in ML dataset
        fare = round(2500 + distance_km * 2.5)
        low = round(fare * 0.80)
        high = round(fare * 1.25)
        prediction_source = "Distance estimate (city not in ML dataset)"

    tier = get_airline_tier(airline_normalized, "IN")

    # OUTPUT - INDIA
    print(f"\n{'=' * 60}")
    print(f"Flight Details:")
    print(f"   Route:       {origin} -> {destination} ({distance_km} km)")
    print(f"   Date:        {date} ({day_names[day_of_week]})")
    print(f"   Departure:   {departure_time}")
    print(f"   Class:       {flight_class}")
    print(f"   Stops:       {stops_label}")
    print(f"   Airline:     {airline_normalized} ({tier} tier)")
    print(f"   Booking:     {days_until_departure} days before departure")
    print(f"   Source:      {prediction_source}")

    print(f"\n{'_' * 60}")
    print(f"Predicted Fare:   Rs.{fare:,}")
    print(f"Fare Range:       Rs.{low:,} - Rs.{high:,}")
    print(f"{'_' * 60}")

    # Dataset prices (actual data)
    if in_ds_route_stats or in_ds_airline_stats:
        print(f"\n{'_' * 60}")
        print(f"DATASET PRICES (actual fares from Indian flight data):")
        print(f"{'_' * 60}")

        if in_ds_route_stats:
            rc = in_ds_route_stats['count']
            print(f"\n   Route {origin} -> {destination} ({flight_class}, all airlines, {rc:,} flights):")
            print(f"      Average:  Rs.{in_ds_route_stats['mean']:,}")
            print(f"      Median:   Rs.{in_ds_route_stats['median']:,}")
            print(f"      Min:      Rs.{in_ds_route_stats['min']:,}")
            print(f"      Max:      Rs.{in_ds_route_stats['max']:,}")

        if in_ds_airline_stats:
            ac = in_ds_airline_stats['count']
            print(f"\n   {airline_normalized} on this route ({ac:,} flights):")
            print(f"      Average:  Rs.{in_ds_airline_stats['mean']:,}")
            print(f"      Median:   Rs.{in_ds_airline_stats['median']:,}")
            print(f"      Min:      Rs.{in_ds_airline_stats['min']:,}")
            print(f"      Max:      Rs.{in_ds_airline_stats['max']:,}")
        elif airline_in_india_dataset and in_ds_route_stats:
            print(f"\n   {airline_normalized}: Does not fly this route in dataset")

        if in_ds_similar_stats:
            print(f"\n   Similar booking window ({dl_bucket_min}-{dl_bucket_max} days, {in_ds_similar_stats['count']:,} flights):")
            print(f"      Average:  Rs.{in_ds_similar_stats['mean']:,}")
            print(f"      Min:      Rs.{in_ds_similar_stats['min']:,}")
            print(f"      Max:      Rs.{in_ds_similar_stats['max']:,}")

        # Accuracy check
        if in_ds_airline_stats or in_ds_route_stats:
            if in_ds_airline_stats:
                benchmark_mean = in_ds_airline_stats["mean"]
                benchmark_label = f"{airline_normalized} avg"
            else:
                benchmark_mean = in_ds_route_stats["mean"]
                benchmark_label = "route avg"

            diff_pct = abs(fare - benchmark_mean) / benchmark_mean * 100
            print(f"\n   Prediction vs Dataset ({benchmark_label}): ", end="")
            if diff_pct <= 10:
                print(f"Rs.{fare:,} vs Rs.{benchmark_mean:,} (within {diff_pct:.1f}% [OK])")
            elif diff_pct <= 25:
                print(f"Rs.{fare:,} vs Rs.{benchmark_mean:,} (off by {diff_pct:.1f}% [WARN])")
            else:
                print(f"Rs.{fare:,} vs Rs.{benchmark_mean:,} (off by {diff_pct:.1f}% [!])")

    # All airlines on this route
    if len(in_route_data) > 0:
        airline_comp = in_route_data.groupby("airline")["price"].agg(["mean", "min", "max", "count"])
        airline_comp = airline_comp.sort_values("mean")
        print(f"\nAll Airlines on {origin} -> {destination} ({flight_class}, from dataset):")
        print(f"   {'Airline':<20} {'Avg':>10} {'Min':>10} {'Max':>10} {'Flights':>8}")
        print(f"   {'_' * 58}")
        for aname, row in airline_comp.iterrows():
            marker = " << YOU" if aname == airline_normalized.lower() else ""
            print(f"   {aname:<20} Rs.{row['mean']:>7,.0f} Rs.{row['min']:>7,.0f} Rs.{row['max']:>7,.0f} {int(row['count']):>8}{marker}")

    print(f"\n{'=' * 60}")
    print(f"\nTips to save money:")
    if days_until_departure < 14:
        print(f"   - Book earlier! Prices are typically cheaper 3-6 weeks out")
    if tier == "premium" and flight_class == "Economy":
        print(f"   - IndiGo/SpiceJet are typically 30-50% cheaper than {airline_normalized}")
    if len(in_route_data) > 0:
        cheapest = in_route_data.groupby("airline")["price"].mean().idxmin()
        cheapest_fare = in_route_data.groupby("airline")["price"].mean().min()
        if cheapest != airline_normalized:
            print(f"   - Cheapest airline on this route: {cheapest} (avg Rs.{cheapest_fare:,.0f})")
    if is_weekend:
        print(f"   - Fly midweek (Tue/Wed) to save 10-20%")
    if 5 <= dep_hour <= 7 or 18 <= dep_hour <= 21:
        print(f"   - Off-peak departures (11am-4pm) are usually cheaper")
    if month in [5, 6, 10] or (month == 11 and day <= 15) or (month == 12 and day >= 20) or (month == 1 and day <= 5):
        print(f"   - Avoid festival/holiday weeks - prices surge 25-50%")

else:
    # US: ML model prediction (trained on your dataset)
    route_te = route_encoding.get(route, global_mean)

    if dep_hour < 6:
        dep_time_bin = "Early_Morning"
    elif dep_hour < 12:
        dep_time_bin = "Morning"
    elif dep_hour < 18:
        dep_time_bin = "Afternoon"
    else:
        dep_time_bin = "Evening_Night"

    arr_minutes_total = dep_hour * 60 + dep_min + scheduled_time
    arr_hour = int((arr_minutes_total // 60) % 24)
    arr_min = int(arr_minutes_total % 60)

    if distance_miles <= 316:
        distance_bin = "(30.999, 316.0]"
    elif distance_miles <= 516:
        distance_bin = "(316.0, 516.0]"
    elif distance_miles <= 757:
        distance_bin = "(516.0, 757.0]"
    elif distance_miles <= 1068:
        distance_bin = "(757.0, 1068.0]"
    else:
        distance_bin = "(1068.0, 2130.0]"

    input_data = {
        "MONTH": month, "DAY": day, "DAY_OF_WEEK": day_of_week,
        "ORIGIN_AIRPORT": origin, "DESTINATION_AIRPORT": destination,
        "SCHEDULED_TIME": scheduled_time, "DISTANCE": distance_miles,
        "AIRLINE": airline_normalized, "DEP_HOUR": dep_hour,
        "ARR_HOUR": arr_hour, "DEP_MIN": dep_min, "ARR_MIN": arr_min,
        "IS_WEEKEND": is_weekend, "DEP_TIME_BIN": dep_time_bin,
        "IS_HOLIDAY_SEASON": is_holiday_season, "DISTANCE_BIN": distance_bin,
        "ROUTE_te": route_te,
    }
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_columns]
    raw_prediction = float(pipeline.predict(df_input)[0])

    # Use the ML model prediction as the fare
    model_fare = max(raw_prediction, 35.0)

    # Apply days_until_departure adjustment (deterministic, conservative)
    if days_until_departure <= 1:
        dept_mult = 1.35
    elif days_until_departure <= 3:
        dept_mult = 1.25
    elif days_until_departure <= 7:
        dept_mult = 1.15
    elif days_until_departure <= 14:
        dept_mult = 1.08
    elif days_until_departure <= 21:
        dept_mult = 1.00
    elif days_until_departure <= 45:
        dept_mult = 0.95
    elif days_until_departure <= 90:
        dept_mult = 0.90
    else:
        dept_mult = 0.85

    fare = model_fare * dept_mult
    fare = round(fare, 2)

    # Anchor US fare to dataset statistics (prevents wild deviations)
    if ds_route_airline_stats:
        anchor = ds_route_airline_stats["mean"]
        fare = round(0.50 * fare + 0.50 * anchor, 2)
    elif ds_route_stats:
        anchor = ds_route_stats["mean"]
        fare = round(0.45 * fare + 0.55 * anchor, 2)

    # Clamp fare within dataset range
    if ds_route_airline_stats:
        fare = max(fare, ds_route_airline_stats["min"] * 0.85)
        fare = min(fare, ds_route_airline_stats["max"] * 1.15)
        fare = round(fare, 2)
    elif ds_route_stats:
        fare = max(fare, ds_route_stats["min"] * 0.85)
        fare = min(fare, ds_route_stats["max"] * 1.15)
        fare = round(fare, 2)

    # Use dataset stats for the fare range if available
    if ds_route_airline_stats:
        low = ds_route_airline_stats["min"]
        high = ds_route_airline_stats["max"]
    elif ds_route_stats:
        low = ds_route_stats["min"]
        high = ds_route_stats["max"]
    else:
        low = round(fare * 0.85, 2)
        high = round(fare * 1.15, 2)

    tier = get_airline_tier(airline_normalized, market="US")

    # OUTPUT
    print(f"\n{'=' * 60}")
    print(f"📋 Flight Details:")
    print(f"   Route:       {origin} → {destination} ({int(distance_miles)} miles)")
    print(f"   Date:        {date} ({day_names[day_of_week]})")
    print(f"   Departure:   {departure_time}")
    print(f"   Airline:     {airline_normalized} ({tier} tier)")
    print(f"   Booking:     {days_until_departure} days before departure")

    print(f"\n{'─' * 60}")
    print(f"💰 ML Predicted Fare:   ${fare:.2f}")
    print(f"📊 Fare Range:          ${low:.2f} — ${high:.2f}")
    print(f"{'─' * 60}")

    print(f"\n📈 Prediction Breakdown:")
    print(f"   ML model base fare:     ${model_fare:.2f}  (from trained pipeline)")
    print(f"   Booking adjustment:     {days_until_departure}d ahead (×{dept_mult:.2f})")
    print(f"   Final predicted fare:   ${fare:.2f}")

    # Dataset prices (actual data)
    print(f"\n{'─' * 60}")
    print(f"📊 DATASET PRICES (actual fares from your data):")
    print(f"{'─' * 60}")

    if ds_route_stats:
        print(f"\n   📍 Route {origin} → {destination}  (all airlines, {ds_route_stats['count']:,} flights):")
        print(f"      Average:  ${ds_route_stats['mean']:.2f}")
        print(f"      Median:   ${ds_route_stats['median']:.2f}")
        print(f"      Min:      ${ds_route_stats['min']:.2f}")
        print(f"      Max:      ${ds_route_stats['max']:.2f}")
    else:
        print(f"\n   📍 Route {origin} → {destination}: No data in dataset")

    if ds_route_airline_stats:
        print(f"\n   ✈️  {airline_normalized} on this route ({ds_route_airline_stats['count']:,} flights):")
        print(f"      Average:  ${ds_route_airline_stats['mean']:.2f}")
        print(f"      Median:   ${ds_route_airline_stats['median']:.2f}")
        print(f"      Min:      ${ds_route_airline_stats['min']:.2f}")
        print(f"      Max:      ${ds_route_airline_stats['max']:.2f}")
    elif airline_in_dataset and ds_route_stats:
        print(f"\n   ✈️  {airline_normalized}: Does not fly this route in dataset")

    if ds_similar_stats:
        wknd_label = "weekends" if is_weekend else "weekdays"
        month_name = date_dt.strftime("%B")
        print(f"\n   📅 Similar conditions ({month_name}, {wknd_label}, {ds_similar_stats['count']:,} flights):")
        print(f"      Average:  ${ds_similar_stats['mean']:.2f}")
        print(f"      Min:      ${ds_similar_stats['min']:.2f}")
        print(f"      Max:      ${ds_similar_stats['max']:.2f}")

    # Accuracy check
    if ds_route_stats:
        diff_pct = abs(fare - ds_route_stats["mean"]) / ds_route_stats["mean"] * 100
        print(f"\n   🎯 Prediction vs Dataset avg: ", end="")
        if diff_pct <= 10:
            print(f"${fare:.2f} vs ${ds_route_stats['mean']:.2f} (within {diff_pct:.1f}% ✅)")
        elif diff_pct <= 25:
            print(f"${fare:.2f} vs ${ds_route_stats['mean']:.2f} (off by {diff_pct:.1f}% ⚠️)")
        else:
            print(f"${fare:.2f} vs ${ds_route_stats['mean']:.2f} (off by {diff_pct:.1f}% ❌)")

    print(f"\n{'=' * 60}")

    # All airlines on this route from dataset
    if ds_route_stats and len(route_data) > 0:
        airline_comparison = route_data.groupby("AIRLINE")["FARE"].agg(["mean", "min", "max", "count"])
        airline_comparison = airline_comparison.sort_values("mean")
        print(f"\n💲 All Airlines on {origin} → {destination} (from dataset):")
        print(f"   {'Airline':<30} {'Avg':>8} {'Min':>8} {'Max':>8} {'Flights':>8}")
        print(f"   {'─' * 62}")
        for aname, row in airline_comparison.iterrows():
            marker = " ◄ YOU" if aname == airline_normalized else ""
            print(f"   {aname:<30} ${row['mean']:>7.2f} ${row['min']:>7.2f} ${row['max']:>7.2f} {int(row['count']):>8}{marker}")

    print(f"\n💡 Tips to save money:")
    if ds_route_stats and len(route_data) > 0:
        cheapest_airline = route_data.groupby("AIRLINE")["FARE"].mean().idxmin()
        cheapest_fare = route_data.groupby("AIRLINE")["FARE"].mean().min()
        if cheapest_airline != airline_normalized:
            print(f"   • Cheapest airline on this route: {cheapest_airline} (avg ${cheapest_fare:.2f})")
    if days_until_departure < 14:
        print(f"   • Book earlier! Prices are typically cheaper 3-8 weeks out")
    if is_weekend:
        print(f"   • Fly midweek (Tue/Wed) to save 10-20%")
    if 6 <= dep_hour <= 8 or 16 <= dep_hour <= 19:
        print(f"   • Off-peak departures (10am-3pm) are usually cheaper")
    if is_holiday_season or month in [6, 7, 8]:
        print(f"   • Travel during off-peak months (Jan-Mar, Sep-Nov) for best deals")

print()
