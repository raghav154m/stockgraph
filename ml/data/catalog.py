"""Product-family catalogue: display names, categories, simulated unit costs (₹) and lead times.

Favorita gives only family names and sales. Costs and lead times are simulated but kept
plausible so the inventory maths produces believable numbers.
"""

from __future__ import annotations

import re

# family -> (display name, category, unit cost in ₹, supplier lead time in days)
FAMILIES: dict[str, tuple[str, str, float, int]] = {
    "AUTOMOTIVE": ("Automotive", "General merchandise", 450.0, 7),
    "BABY CARE": ("Baby Care", "Personal care", 320.0, 5),
    "BEAUTY": ("Beauty", "Personal care", 280.0, 5),
    "BEVERAGES": ("Beverages", "Beverages", 60.0, 3),
    "BOOKS": ("Books", "General merchandise", 250.0, 10),
    "BREAD/BAKERY": ("Bread & Bakery", "Fresh", 45.0, 1),
    "CELEBRATION": ("Celebration", "General merchandise", 180.0, 7),
    "CLEANING": ("Cleaning", "Household", 120.0, 5),
    "DAIRY": ("Dairy", "Fresh", 55.0, 1),
    "DELI": ("Deli", "Fresh", 150.0, 2),
    "EGGS": ("Eggs", "Fresh", 7.0, 2),
    "FROZEN FOODS": ("Frozen Foods", "Grocery", 210.0, 4),
    "GROCERY I": ("Grocery (Staples)", "Grocery", 90.0, 3),
    "GROCERY II": ("Grocery (Specialty)", "Grocery", 160.0, 5),
    "HARDWARE": ("Hardware", "General merchandise", 380.0, 10),
    "HOME AND KITCHEN I": ("Home & Kitchen (Essentials)", "Household", 350.0, 7),
    "HOME AND KITCHEN II": ("Home & Kitchen (Premium)", "Household", 650.0, 10),
    "HOME APPLIANCES": ("Home Appliances", "General merchandise", 2500.0, 14),
    "HOME CARE": ("Home Care", "Household", 140.0, 5),
    "LADIESWEAR": ("Ladieswear", "Apparel", 700.0, 10),
    "LAWN AND GARDEN": ("Lawn & Garden", "General merchandise", 300.0, 10),
    "LINGERIE": ("Lingerie", "Apparel", 400.0, 10),
    "LIQUOR,WINE,BEER": ("Liquor, Wine & Beer", "Beverages", 550.0, 5),
    "MAGAZINES": ("Magazines", "General merchandise", 80.0, 7),
    "MEATS": ("Meats", "Fresh", 380.0, 2),
    "PERSONAL CARE": ("Personal Care", "Personal care", 150.0, 5),
    "PET SUPPLIES": ("Pet Supplies", "General merchandise", 300.0, 7),
    "PLAYERS AND ELECTRONICS": ("Players & Electronics", "General merchandise", 1800.0, 14),
    "POULTRY": ("Poultry", "Fresh", 240.0, 2),
    "PREPARED FOODS": ("Prepared Foods", "Fresh", 120.0, 1),
    "PRODUCE": ("Produce", "Fresh", 40.0, 1),
    "SCHOOL AND OFFICE SUPPLIES": ("School & Office Supplies", "General merchandise", 90.0, 7),
    "SEAFOOD": ("Seafood", "Fresh", 420.0, 2),
}

DEFAULT = ("", "Other", 100.0, 5)


def product_id(family: str) -> str:
    """Stable, URL-friendly id: 'LIQUOR,WINE,BEER' -> 'liquor-wine-beer'."""
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")


def product_row(family: str) -> dict:
    name, category, cost, lead = FAMILIES.get(family, DEFAULT)
    return {
        "product_id": product_id(family),
        "name": name or family.title(),
        "category": category,
        "unit_cost": cost,
        "lead_time_days": lead,
    }
