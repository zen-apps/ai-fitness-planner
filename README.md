# ai-fitness-planner

download usda json branded foods data 
@workout.get("/load_usda_data/")

load data into mongo
@workout.post("/import_usda_data/")

search by product 
@workout.get("/search_nutrition/")
{
  "query": "black label",
  "results_found": 1,
  "results": [
    {
      "fdc_id": 2070292,
      "description": "HORMEL, BLACK LABEL, BROWN SUGAR THICK CUT BACON",
      "brand_owner": "Hormel Foods Corporation ",
      "brand_name": "HORMEL",
      "food_class": "Branded",
      "food_category": null,
      "gtin_upc": "037600314060",
      "ingredients": "CURED WITH WATER, SALT, BROWN SUGAR, SODIUM PHOSPHATE, SODIUM ERYTHORBATE, NATURAL FLAVOR (WATER, NATURAL FLAVORS), SODIUM NITRITE.",
      "serving_size": 24,
      "serving_size_unit": "g",
      "household_serving_fulltext": "2 PAN-FRIED SLICES",
      "modified_date": "7/11/2020",
      "available_date": "7/11/2020",
      "market_country": "United States",
      "discontinued_date": null,
      "preparation_state_code": null,
      "trade_channel": null,
      "short_description": null,
      "nutrition_enhanced": {
        "serving_info": {
          "serving_size_g": 24,
          "serving_description": "2 PAN-FRIED SLICES",
          "multiplier_to_100g": 4.1667
        },
        "per_serving": {
          "energy_kcal": 458,
          "protein_g": 33.3,
          "total_fat_g": 33.3,
          "carbs_g": 4.17,
          "fiber_g": 0,
          "sugars_g": 4.17,
          "sodium_mg": 1960,
          "cholesterol_mg": 104,
          "saturated_fat_g": 12.5,
          "trans_fat_g": 0
        },
        "per_100g": {
          "energy_kcal": 1908.33,
          "protein_g": 138.75,
          "total_fat_g": 138.75,
          "carbs_g": 17.38,
          "fiber_g": 0,
          "sugars_g": 17.38,
          "sodium_mg": 8166.67,
          "cholesterol_mg": 433.33,
          "saturated_fat_g": 52.08,
          "trans_fat_g": 0
        },
        "label_nutrients_enhanced": {
          "calories": 110,
          "fat_g": 7.99,
          "saturated_fat_g": 3,
          "trans_fat_g": 0,
          "cholesterol_mg": 25,
          "sodium_mg": 470,
          "carbs_g": 1,
          "fiber_g": 0,
          "sugars_g": 1,
          "protein_g": 7.99
        },
        "nutrition_density_score": 7.27,
        "macro_breakdown": {
          "protein_percent": 29.6,
          "fat_percent": 66.7,
          "carbs_percent": 3.7,
          "total_macro_kcal": 1873.3,
          "calories_from_protein": 555,
          "calories_from_fat": 1248.8,
          "calories_from_carbs": 69.5,
          "macro_categories": [
            "high_fat"
          ],
          "primary_macro_category": "high_fat",
          "is_high_protein": false,
          "is_high_fat": true,
          "is_high_carb": false,
          "is_balanced": false
        }
      },
      "food_nutrients": [
        {
          "type": "FoodNutrient",
          "id": 25472934,
          "nutrient": {
            "id": 1003,
            "number": "203",
            "name": "Protein",
            "rank": 600,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 33.3
        },
        {
          "type": "FoodNutrient",
          "id": 25472935,
          "nutrient": {
            "id": 1004,
            "number": "204",
            "name": "Total lipid (fat)",
            "rank": 800,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 33.3
        },
        {
          "type": "FoodNutrient",
          "id": 25472936,
          "nutrient": {
            "id": 1005,
            "number": "205",
            "name": "Carbohydrate, by difference",
            "rank": 1110,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 4.17
        },
        {
          "type": "FoodNutrient",
          "id": 25472937,
          "nutrient": {
            "id": 1008,
            "number": "208",
            "name": "Energy",
            "rank": 300,
            "unitName": "kcal"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 458
        },
        {
          "type": "FoodNutrient",
          "id": 25472938,
          "nutrient": {
            "id": 2000,
            "number": "269",
            "name": "Total Sugars",
            "rank": 1510,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 4.17
        },
        {
          "type": "FoodNutrient",
          "id": 25472939,
          "nutrient": {
            "id": 1079,
            "number": "291",
            "name": "Fiber, total dietary",
            "rank": 1200,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCD",
            "description": "Calculated from a daily value percentage per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 25472940,
          "nutrient": {
            "id": 1087,
            "number": "301",
            "name": "Calcium, Ca",
            "rank": 5300,
            "unitName": "mg"
          },
          "foodNutrientDerivation": {
            "code": "LCCD",
            "description": "Calculated from a daily value percentage per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 25472941,
          "nutrient": {
            "id": 1089,
            "number": "303",
            "name": "Iron, Fe",
            "rank": 5400,
            "unitName": "mg"
          },
          "foodNutrientDerivation": {
            "code": "LCCD",
            "description": "Calculated from a daily value percentage per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 25472942,
          "nutrient": {
            "id": 1093,
            "number": "307",
            "name": "Sodium, Na",
            "rank": 5800,
            "unitName": "mg"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 1960
        },
        {
          "type": "FoodNutrient",
          "id": 25472943,
          "nutrient": {
            "id": 1104,
            "number": "318",
            "name": "Vitamin A, IU",
            "rank": 7500,
            "unitName": "IU"
          },
          "foodNutrientDerivation": {
            "code": "LCCD",
            "description": "Calculated from a daily value percentage per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 25472944,
          "nutrient": {
            "id": 1162,
            "number": "401",
            "name": "Vitamin C, total ascorbic acid",
            "rank": 6300,
            "unitName": "mg"
          },
          "foodNutrientDerivation": {
            "code": "LCCD",
            "description": "Calculated from a daily value percentage per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 25472945,
          "nutrient": {
            "id": 1253,
            "number": "601",
            "name": "Cholesterol",
            "rank": 15700,
            "unitName": "mg"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 104
        },
        {
          "type": "FoodNutrient",
          "id": 25472946,
          "nutrient": {
            "id": 1257,
            "number": "605",
            "name": "Fatty acids, total trans",
            "rank": 15400,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 25472947,
          "nutrient": {
            "id": 1258,
            "number": "606",
            "name": "Fatty acids, total saturated",
            "rank": 9700,
            "unitName": "g"
          },
          "foodNutrientDerivation": {
            "code": "LCCS",
            "description": "Calculated from value per serving size measure",
            "foodNutrientSource": {
              "id": 9,
              "code": "12",
              "description": "Manufacturer's analytical; partial documentation"
            }
          },
          "amount": 12.5
        }
      ],
      "food_attributes": [],
      "food_attribute_types": [],
      "food_version_ids": []
    }
  ]
}