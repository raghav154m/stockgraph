"""Telangana places used to relabel the Favorita stores.

Favorita stores sit in Ecuadorian cities. We map each original city to a Telangana city,
largest to largest, so stores from the same original city stay together and the sales
patterns and graph structure stay consistent.

Coordinates are approximate (good enough for a map), in decimal degrees.
"""

from __future__ import annotations

import math

# Hyderabad areas, used for the city with the most stores.
HYDERABAD_AREAS: list[tuple[str, float, float]] = [
    ("Gachibowli", 17.4401, 78.3489),
    ("Madhapur", 17.4483, 78.3915),
    ("Kukatpally", 17.4849, 78.4138),
    ("Ameerpet", 17.4375, 78.4482),
    ("Secunderabad", 17.4399, 78.4983),
    ("Dilsukhnagar", 17.3688, 78.5247),
    ("LB Nagar", 17.3457, 78.5522),
    ("Banjara Hills", 17.4156, 78.4347),
    ("Jubilee Hills", 17.4325, 78.4073),
    ("Miyapur", 17.4968, 78.3614),
    ("Uppal", 17.4058, 78.5591),
    ("Mehdipatnam", 17.3959, 78.4369),
    ("Begumpet", 17.4447, 78.4664),
    ("Kondapur", 17.4699, 78.3577),
    ("Himayatnagar", 17.4020, 78.4870),
    ("Tarnaka", 17.4276, 78.5300),
    ("Malkajgiri", 17.4505, 78.5265),
    ("Attapur", 17.3654, 78.4291),
    ("Manikonda", 17.4050, 78.3866),
    ("Kompally", 17.5360, 78.4865),
    ("Nampally", 17.3871, 78.4669),
    ("Abids", 17.3920, 78.4760),
]

# Areas for the second-largest city (Warangal tri-city).
WARANGAL_AREAS: list[tuple[str, float, float]] = [
    ("Hanamkonda", 18.0072, 79.5584),
    ("Kazipet", 17.9746, 79.5004),
    ("Warangal Fort Road", 17.9575, 79.6140),
    ("Hunter Road", 17.9870, 79.5720),
    ("Subedari", 18.0010, 79.5650),
    ("Naimnagar", 18.0160, 79.5480),
    ("Station Road", 17.9700, 79.6000),
    ("Kishanpura", 18.0200, 79.5400),
    ("Matwada", 17.9650, 79.5900),
    ("Balasamudram", 18.0050, 79.5530),
]

# Remaining cities, roughly in order of size. Each gets its city-centre coordinate.
OTHER_CITIES: list[tuple[str, float, float]] = [
    ("Nizamabad", 18.6725, 78.0941),
    ("Karimnagar", 18.4386, 79.1288),
    ("Khammam", 17.2473, 80.1514),
    ("Ramagundam", 18.7550, 79.4740),
    ("Mahbubnagar", 16.7488, 78.0035),
    ("Nalgonda", 17.0575, 79.2671),
    ("Adilabad", 19.6641, 78.5320),
    ("Suryapet", 17.1405, 79.6236),
    ("Siddipet", 18.1018, 78.8520),
    ("Miryalaguda", 16.8722, 79.5625),
    ("Jagtial", 18.7909, 78.9119),
    ("Mancherial", 18.8714, 79.4443),
    ("Kothagudem", 17.5500, 80.6200),
    ("Nirmal", 19.0960, 78.3440),
    ("Kamareddy", 18.3200, 78.3400),
    ("Sangareddy", 17.6140, 78.0816),
    ("Zaheerabad", 17.6815, 77.6074),
    ("Bhongir", 17.5100, 78.8900),
    ("Wanaparthy", 16.3620, 78.0620),
    ("Medak", 18.0450, 78.2600),
    ("Vikarabad", 17.3380, 77.9040),
    ("Jangaon", 17.7240, 79.1520),
]

# Store names for small cities with more than one store.
LOCAL_AREAS = ["Main Road", "Bus Stand Road", "Market Area", "Station Road", "Collectorate Road"]

# Simulated regional warehouses (distribution centres).
WAREHOUSES: list[dict] = [
    {"location_id": "WH-SHM", "name": "Shamshabad DC", "city": "Shamshabad",
     "region": "South Telangana", "latitude": 17.2403, "longitude": 78.4294},
    {"location_id": "WH-MDC", "name": "Medchal DC", "city": "Medchal",
     "region": "North Telangana", "latitude": 17.6297, "longitude": 78.4814},
    {"location_id": "WH-WGL", "name": "Warangal DC", "city": "Warangal",
     "region": "East Telangana", "latitude": 17.9469, "longitude": 79.5600},
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_warehouse(lat: float, lon: float) -> dict:
    return min(WAREHOUSES, key=lambda w: haversine_km(lat, lon, w["latitude"], w["longitude"]))


def assign_city_names(store_counts: dict[str, int]) -> dict[str, tuple[str, list[tuple[str, float, float]]]]:
    """Map each original city to a Telangana city and a list of store sites.

    Args:
        store_counts: original city -> number of stores in it.

    Returns:
        original city -> (Telangana city, [(site name, lat, lon), ...] one per store).
    """
    # Largest first; ties broken by name so the mapping is stable across runs.
    ordered = sorted(store_counts, key=lambda c: (-store_counts[c], c))
    if len(ordered) > 2 + len(OTHER_CITIES):
        raise ValueError(f"Need more Telangana cities: {len(ordered)} original cities found")

    mapping: dict[str, tuple[str, list[tuple[str, float, float]]]] = {}
    for rank, original in enumerate(ordered):
        n = store_counts[original]
        if rank == 0:
            city, pool = "Hyderabad", HYDERABAD_AREAS
        elif rank == 1:
            city, pool = "Warangal", WARANGAL_AREAS
        else:
            city, lat, lon = OTHER_CITIES[rank - 2]
            # Small, fixed offsets so multiple stores in one town don't sit on one point.
            pool = [
                (LOCAL_AREAS[i % len(LOCAL_AREAS)],
                 lat + 0.008 * math.cos(i * 2.1),
                 lon + 0.008 * math.sin(i * 2.1))
                for i in range(n)
            ]
            if n == 1:
                pool = [(city, lat, lon)]
        if n > len(pool):
            raise ValueError(f"{original} has {n} stores but only {len(pool)} {city} sites are defined")
        mapping[original] = (city, pool[:n])
    return mapping
