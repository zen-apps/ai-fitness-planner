#!/usr/bin/env python3
"""
Generate realistic sample nutrition data for quick demo setup.
This creates a curated subset of nutrition data that covers all major food categories
without requiring the full USDA dataset download.
"""

import json
import os
import random
from typing import Dict, List, Any


def create_nutrition_sample() -> List[Dict[str, Any]]:
    """Generate realistic nutrition data for quick demos"""

    # Define food categories with representative foods
    sample_foods = []

    # High-protein foods
    protein_foods = [
        {
            "fdc_id": 10001,
            "description": "Chicken breast, grilled, skinless",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Poultry Products",
            "gtin_upc": "000000000001",
            "ingredients": "Chicken breast",
            "serving_size": 100,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 piece (100g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 165,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 31.0,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 3.6,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 0,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 74,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 85,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 1.0,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        },
        {
            "fdc_id": 10002,
            "description": "Greek yogurt, plain, nonfat",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Dairy Products",
            "gtin_upc": "000000000002",
            "ingredients": "Cultured nonfat milk",
            "serving_size": 170,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 container (170g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 100,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 17.0,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 0.4,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 6.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 6.0,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 56,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 5,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.1,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        },
    ]

    # Carbohydrate-rich foods
    carb_foods = [
        {
            "fdc_id": 10003,
            "description": "Brown rice, cooked",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Cereal Grains and Pasta",
            "gtin_upc": "000000000003",
            "ingredients": "Brown rice, water",
            "serving_size": 150,
            "serving_size_unit": "g",
            "household_serving_fulltext": "3/4 cup cooked (150g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 216,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 5.0,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 1.8,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 45.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 3.5,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 0.7,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 10,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.4,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        },
        {
            "fdc_id": 10004,
            "description": "Whole wheat bread",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Baked Products",
            "gtin_upc": "000000000004",
            "ingredients": "Whole wheat flour, water, yeast, salt",
            "serving_size": 28,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 slice (28g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 69,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 3.6,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 1.2,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 12.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 1.9,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 1.4,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 144,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.3,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        },
    ]

    # Healthy fats
    fat_foods = [
        {
            "fdc_id": 10005,
            "description": "Avocado, raw",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Fruits and Fruit Juices",
            "gtin_upc": "000000000005",
            "ingredients": "Avocado",
            "serving_size": 150,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 medium avocado (150g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 234,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 2.9,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 21.4,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 12.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 10.0,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 1.0,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 10,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 3.1,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        }
    ]

    # Vegetables
    vegetable_foods = [
        {
            "fdc_id": 10006,
            "description": "Broccoli, raw",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Vegetables and Vegetable Products",
            "gtin_upc": "000000000006",
            "ingredients": "Broccoli",
            "serving_size": 100,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 cup chopped (100g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 34,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 2.8,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 0.4,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 7.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 2.6,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 1.5,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 33,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.1,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        },
        {
            "fdc_id": 10007,
            "description": "Sweet potato, baked with skin",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Vegetables and Vegetable Products",
            "gtin_upc": "000000000007",
            "ingredients": "Sweet potato",
            "serving_size": 130,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 medium sweet potato (130g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 112,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 2.0,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 0.1,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 26.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 3.9,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 5.4,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 6,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.0,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        },
    ]

    # Fruits
    fruit_foods = [
        {
            "fdc_id": 10008,
            "description": "Banana, fresh",
            "brand_owner": "Generic",
            "brand_name": "GENERIC",
            "food_class": "Branded",
            "food_category": "Fruits and Fruit Juices",
            "gtin_upc": "000000000008",
            "ingredients": "Banana",
            "serving_size": 120,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1 medium banana (120g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 105,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 1.3,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 0.4,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 27.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 3.1,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 14.4,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 1,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.1,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        }
    ]

    # Popular branded foods for realism
    branded_foods = [
        {
            "fdc_id": 10009,
            "description": "QUAKER, OATS, OLD FASHIONED",
            "brand_owner": "The Quaker Oats Company",
            "brand_name": "QUAKER",
            "food_class": "Branded",
            "food_category": "Breakfast Cereals",
            "gtin_upc": "030000012345",
            "ingredients": "100% WHOLE GRAIN ROLLED OATS",
            "serving_size": 40,
            "serving_size_unit": "g",
            "household_serving_fulltext": "1/2 cup dry oats (40g)",
            "modified_date": "2024-01-01",
            "available_date": "2024-01-01",
            "market_country": "United States",
            "food_nutrients": [
                {
                    "nutrient": {"id": 1008, "name": "Energy", "unitName": "kcal"},
                    "amount": 150,
                },
                {
                    "nutrient": {"id": 1003, "name": "Protein", "unitName": "g"},
                    "amount": 5.0,
                },
                {
                    "nutrient": {
                        "id": 1004,
                        "name": "Total lipid (fat)",
                        "unitName": "g",
                    },
                    "amount": 3.0,
                },
                {
                    "nutrient": {
                        "id": 1005,
                        "name": "Carbohydrate, by difference",
                        "unitName": "g",
                    },
                    "amount": 27.0,
                },
                {
                    "nutrient": {
                        "id": 1079,
                        "name": "Fiber, total dietary",
                        "unitName": "g",
                    },
                    "amount": 4.0,
                },
                {
                    "nutrient": {"id": 2000, "name": "Total Sugars", "unitName": "g"},
                    "amount": 1.0,
                },
                {
                    "nutrient": {"id": 1093, "name": "Sodium, Na", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {"id": 1253, "name": "Cholesterol", "unitName": "mg"},
                    "amount": 0,
                },
                {
                    "nutrient": {
                        "id": 1258,
                        "name": "Fatty acids, total saturated",
                        "unitName": "g",
                    },
                    "amount": 0.5,
                },
                {
                    "nutrient": {
                        "id": 1257,
                        "name": "Fatty acids, total trans",
                        "unitName": "g",
                    },
                    "amount": 0,
                },
            ],
        }
    ]

    # Combine all food categories
    all_sample_foods = (
        protein_foods
        + carb_foods
        + fat_foods
        + vegetable_foods
        + fruit_foods
        + branded_foods
    )

    # Apply enhanced nutrition calculations to each food
    for food in all_sample_foods:
        food = calculate_per_100g_values(food)

    return all_sample_foods


def calculate_per_100g_values(food_item: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate per-100g nutrition values and add enhanced structure (matches existing system)"""
    serving_size = food_item.get("serving_size", 100)

    if not serving_size or serving_size <= 0:
        serving_size = 100

    multiplier = 100 / serving_size
    food_nutrients = food_item.get("food_nutrients", [])

    def extract_nutrient_by_id(food_nutrients, nutrient_id):
        """Extract specific nutrient amount by ID"""
        for nutrient in food_nutrients:
            if nutrient.get("nutrient", {}).get("id") == nutrient_id:
                return nutrient.get("amount", 0)
        return 0

    nutrient_map = {
        1008: "energy_kcal",
        1003: "protein_g",
        1004: "total_fat_g",
        1005: "carbs_g",
        1079: "fiber_g",
        2000: "sugars_g",
        1093: "sodium_mg",
        1253: "cholesterol_mg",
        1258: "saturated_fat_g",
        1257: "trans_fat_g",
    }

    per_serving = {}
    per_100g = {}

    for nutrient_id, nutrient_name in nutrient_map.items():
        amount = extract_nutrient_by_id(food_nutrients, nutrient_id)
        per_serving[nutrient_name] = amount
        per_100g[nutrient_name] = round(amount * multiplier, 2)

    def calculate_nutrition_density_score(per_100g):
        """Calculate a nutrition density score"""
        try:
            protein = per_100g.get("protein_g", 0)
            fiber = per_100g.get("fiber_g", 0)
            calories = per_100g.get("energy_kcal", 1)

            if calories > 0:
                return round((protein + fiber) / calories * 100, 2)
            return 0
        except:
            return 0

    def calculate_macro_breakdown(per_100g):
        """Calculate macronutrient percentages and categorization"""
        try:
            protein_g = per_100g.get("protein_g", 0)
            fat_g = per_100g.get("total_fat_g", 0)
            carbs_g = per_100g.get("carbs_g", 0)

            calories_from_protein = protein_g * 4
            calories_from_fat = fat_g * 9
            calories_from_carbs = carbs_g * 4
            total_calculated_kcal = (
                calories_from_protein + calories_from_fat + calories_from_carbs
            )

            if total_calculated_kcal > 0:
                pct_protein = (calories_from_protein / total_calculated_kcal) * 100
                pct_fat = (calories_from_fat / total_calculated_kcal) * 100
                pct_carbs = (calories_from_carbs / total_calculated_kcal) * 100

                macro_categories = []
                primary_macro = "balanced"

                if pct_protein >= 40:
                    macro_categories.append("high_protein")
                    primary_macro = "high_protein"
                if pct_fat >= 40:
                    macro_categories.append("high_fat")
                    primary_macro = "high_fat"
                if pct_carbs >= 40:
                    macro_categories.append("high_carb")
                    primary_macro = "high_carb"

                if len(macro_categories) > 1:
                    max_pct = max(pct_protein, pct_fat, pct_carbs)
                    if max_pct == pct_protein:
                        primary_macro = "high_protein"
                    elif max_pct == pct_fat:
                        primary_macro = "high_fat"
                    else:
                        primary_macro = "high_carb"

                return {
                    "protein_percent": round(pct_protein, 1),
                    "fat_percent": round(pct_fat, 1),
                    "carbs_percent": round(pct_carbs, 1),
                    "total_macro_kcal": round(total_calculated_kcal, 1),
                    "calories_from_protein": round(calories_from_protein, 1),
                    "calories_from_fat": round(calories_from_fat, 1),
                    "calories_from_carbs": round(calories_from_carbs, 1),
                    "macro_categories": macro_categories,
                    "primary_macro_category": primary_macro,
                    "is_high_protein": pct_protein >= 40,
                    "is_high_fat": pct_fat >= 40,
                    "is_high_carb": pct_carbs >= 40,
                    "is_balanced": len(macro_categories) == 0,
                }

            return {
                "protein_percent": 0,
                "fat_percent": 0,
                "carbs_percent": 0,
                "total_macro_kcal": 0,
                "calories_from_protein": 0,
                "calories_from_fat": 0,
                "calories_from_carbs": 0,
                "macro_categories": [],
                "primary_macro_category": "unknown",
                "is_high_protein": False,
                "is_high_fat": False,
                "is_high_carb": False,
                "is_balanced": False,
            }
        except Exception as e:
            return {
                "protein_percent": 0,
                "fat_percent": 0,
                "carbs_percent": 0,
                "total_macro_kcal": 0,
                "calories_from_protein": 0,
                "calories_from_fat": 0,
                "calories_from_carbs": 0,
                "macro_categories": [],
                "primary_macro_category": "unknown",
                "is_high_protein": False,
                "is_high_fat": False,
                "is_high_carb": False,
                "is_balanced": False,
            }

    food_item["nutrition_enhanced"] = {
        "serving_info": {
            "serving_size_g": serving_size,
            "serving_description": food_item.get("household_serving_fulltext", ""),
            "multiplier_to_100g": round(multiplier, 4),
        },
        "per_serving": per_serving,
        "per_100g": per_100g,
        "label_nutrients_enhanced": {},
        "nutrition_density_score": calculate_nutrition_density_score(per_100g),
        "macro_breakdown": calculate_macro_breakdown(per_100g),
    }

    return food_item


def save_sample_data(output_path: str = "./data/sample_nutrition_data.json"):
    """Generate and save sample nutrition data to file"""

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate sample foods
    sample_foods = create_nutrition_sample()

    # Create structure that matches USDA JSON format
    sample_data = {"BrandedFoods": sample_foods}

    # Save to file
    with open(output_path, "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"Generated {len(sample_foods)} sample foods")
    print(f"Saved sample data to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    output_file = save_sample_data()
    print(f"\nSample nutrition data generated successfully!")
    print(f"Location: {output_file}")
    print("\nTo use this data:")
    print("1. Run: python scripts/setup_database.py --mode=demo")
    print("2. Or use: make setup-demo")
