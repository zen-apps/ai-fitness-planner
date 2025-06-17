# ai-fitness-planner

download usda json branded foods data 
@workout.get("/load_usda_data/")

load data into mongo
@workout.post("/import_usda_data/")

search by product 
@workout.get("/search_nutrition/")
{
  "query": "coca",
  "results_found": 1,
  "results": [
    {
      "fdc_id": 653612,
      "description": "COCA-COLA WITH CHERRY FLAVOR, CHERRY",
      "brand_owner": "COKE CHERRY",
      "brand_name": null,
      "food_class": "Branded",
      "food_category": null,
      "gtin_upc": "04915001",
      "ingredients": "CARBONATED WATER, HIGH FRUCTOSE CORN SYRUP, CARAMEL COLOR, PHOSPHORIC ACID, NATURAL FLAVORS, CAFFEINE",
      "serving_size": 253,
      "serving_size_unit": "ml",
      "household_serving_fulltext": "8.55 OZA",
      "modified_date": "10/3/2019",
      "available_date": "10/3/2019",
      "market_country": "United States",
      "discontinued_date": null,
      "preparation_state_code": null,
      "trade_channel": null,
      "short_description": null,
      "nutrition_enhanced": {
        "serving_info": {
          "serving_size_g": 253,
          "serving_description": "8.55 OZA",
          "multiplier_to_100g": 0.3953
        },
        "per_serving": {
          "energy_kcal": 43,
          "protein_g": 0,
          "total_fat_g": 0,
          "carbs_g": 11.9,
          "fiber_g": 0,
          "sugars_g": 11.9,
          "sodium_mg": 10,
          "cholesterol_mg": 0,
          "saturated_fat_g": 0,
          "trans_fat_g": 0
        },
        "per_100g": {
          "energy_kcal": 17,
          "protein_g": 0,
          "total_fat_g": 0,
          "carbs_g": 4.7,
          "fiber_g": 0,
          "sugars_g": 4.7,
          "sodium_mg": 3.95,
          "cholesterol_mg": 0,
          "saturated_fat_g": 0,
          "trans_fat_g": 0
        },
        "label_nutrients_enhanced": {
          "calories": 109,
          "fat_g": 0,
          "sodium_mg": 25.3,
          "carbs_g": 30.1,
          "sugars_g": 30.1,
          "protein_g": 0
        },
        "nutrition_density_score": 0,
        "macro_breakdown": {
          "protein_percent": 0,
          "fat_percent": 0,
          "carbs_percent": 100,
          "total_macro_kcal": 18.8,
          "calories_from_protein": 0,
          "calories_from_fat": 0,
          "calories_from_carbs": 18.8,
          "macro_categories": [
            "high_carb"
          ],
          "primary_macro_category": "high_carb",
          "is_high_protein": false,
          "is_high_fat": false,
          "is_high_carb": true,
          "is_balanced": false
        }
      },
      "food_nutrients": [
        {
          "type": "FoodNutrient",
          "id": 7095744,
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
          "amount": 0
        },
        {
          "type": "FoodNutrient",
          "id": 7095746,
          "nutrient": {
            "id": 1004,
            "number": "204",
            "name": "Total lipid (fat)",
            "rank": 800,
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
          "id": 7095748,
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
          "amount": 11.9
        },
        {
          "type": "FoodNutrient",
          "id": 7095750,
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
          "amount": 43
        },
        {
          "type": "FoodNutrient",
          "id": 7095752,
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
          "amount": 11.9
        },
        {
          "type": "FoodNutrient",
          "id": 7095754,
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
          "amount": 10
        }
      ],
      "food_attributes": [
        {
          "id": 115276,
          "name": "Description",
          "value": "2",
          "foodAttributeType": {
            "id": 998,
            "name": "Update Log",
            "description": "Changes that were made to this food"
          }
        }
      ],
      "food_attribute_types": [],
      "food_version_ids": []
    }
  ]
}

returns doc count and fields 
@workout.get("/check_usda_data/")
{
  "status": "data_exists",
  "collection": "branded_foods",
  "document_count": 452998,
  "sample_fields": [
    "_id",
    "foodClass",
    "description",
    "foodNutrients",
    "foodAttributes",
    "modifiedDate",
    "availableDate",
    "marketCountry",
    "brandOwner",
    "gtinUpc",
    "dataSource",
    "ingredients",
    "servingSize",
    "servingSizeUnit",
    "householdServingFullText",
    "labelNutrients",
    "tradeChannels",
    "microbes",
    "brandedFoodCategory",
    "dataType",
    "fdcId",
    "publicationDate",
    "foodUpdateLog"
  ],
  "message": "Found 452,998 branded food documents"
}