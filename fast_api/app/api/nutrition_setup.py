import os
import json
import logging
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import requests
import zipfile
from pathlib import Path
import time
from tqdm import tqdm
import ijson
from decimal import Decimal
from pymongo import MongoClient
import pymongo

from app.api.helpers.usda_branded_foods import (
    download_usda_branded_foods,
    verify_json_structure,
)

# Set up logging with a more specific name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("workout_optimization")

nutrition_setup = APIRouter()


@nutrition_setup.get("/download_raw_usda_data/")
async def download_raw_usda_data():
    """Load USDA Branded Foods data into PostgreSQL database"""

    # Step 1: Download and extract the USDA Branded Foods data
    json_file = download_usda_branded_foods()
    if not json_file:
        error_msg = "Failed to download USDA Branded Foods data."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # Step 2: Verify the JSON structure
    if not verify_json_structure(json_file):
        error_msg = "USDA Branded Foods JSON structure is invalid."
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # is the json file path valid return true or false
    if not os.path.isfile(json_file):
        error_msg = f"JSON file not found at path: {json_file}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    else:
        logger.info(f"JSON file found at path: {json_file}")


# Fixed MongoDB connection test endpoint
@nutrition_setup.get("/test_mongo_db/")
async def test_mongo_db():
    """Test MongoDB connection with improved error handling"""
    from pymongo import MongoClient, errors

    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_db_name = os.getenv("MONGO_DB_NAME")

    mongo_hosts_to_try = [
        "mongodb_ai_fitness_planner",  # Docker service name
        "localhost",  # If running locally
    ]

    for host in mongo_hosts_to_try:
        try:
            logger.info(f"Attempting to connect to MongoDB at host: {host}")

            # Fixed URI format - connect to admin database for authentication
            if host == "mongodb_ai_fitness_planner":
                port = 27017  # Internal Docker port
            else:
                port = 27019  # External mapped port

            mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{host}:{port}/admin"

            logger.info(
                f"Using MongoDB URI: mongodb://{mongo_user}:***@{host}:{port}/admin"
            )

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)

            # Test the connection
            client.admin.command("ping")

            # Access your specific database
            db = client[mongo_db_name] if mongo_db_name else client["usda_nutrition"]
            collection_names = db.list_collection_names()

            # Test if we can access existing collections
            existing_collections = []
            for collection_name in collection_names[:5]:  # Limit to first 5
                try:
                    count = db[collection_name].count_documents({})
                    existing_collections.append(
                        {"name": collection_name, "document_count": count}
                    )
                except Exception as e:
                    existing_collections.append(
                        {"name": collection_name, "error": str(e)}
                    )

            client.close()

            logger.info(f"Successfully connected to MongoDB at host: {host}")
            return {
                "status": "success",
                "connected_host": host,
                "port": port,
                "database_name": mongo_db_name or "usda_nutrition",
                "total_collections": len(collection_names),
                "collections_sample": existing_collections,
                "environment_vars": {
                    "MONGO_USER": mongo_user,
                    "MONGO_PASSWORD": "***" if mongo_password else None,
                    "MONGO_DB_NAME": mongo_db_name,
                },
            }

        except errors.ServerSelectionTimeoutError as err:
            logger.warning(
                f"Server selection timeout for MongoDB at host {host}: {str(err)}"
            )
            continue
        except errors.OperationFailure as err:
            logger.warning(
                f"Authentication failed for MongoDB at host {host}: {str(err)}"
            )
            continue
        except Exception as err:
            logger.warning(
                f"Unexpected error connecting to MongoDB at host {host}: {str(err)}"
            )
            continue

    # If all hosts failed
    error_msg = (
        f"Failed to connect to MongoDB. Tried hosts: {', '.join(mongo_hosts_to_try)}"
    )
    logger.error(error_msg)
    raise HTTPException(status_code=500, detail=error_msg)


# Also add a simpler connection helper function
def get_mongo_client():
    """Get MongoDB client connection"""
    from pymongo import MongoClient

    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")

    # Try Docker service name first, then localhost
    try:
        # Docker internal connection
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@mongodb_ai_fitness_planner:27017/admin"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # Test connection
        return client
    except:
        # Fallback to localhost
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:27019/admin"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # Test connection
        return client


@nutrition_setup.get("/search_nutrition/")
async def search_nutrition(query: str = "coca cola", limit: int = 1):
    """Search nutrition data to test MongoDB functionality"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]

        # Search by brand owner or description
        search_query = {
            "$or": [
                {"brandOwner": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
            ]
        }

        results = list(db.branded_foods.find(search_query).limit(limit))

        # Format results for display
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "fdc_id": result.get("fdcId"),
                    "description": result.get("description"),
                    "brand_owner": result.get("brandOwner"),
                    "brand_name": result.get("brandName"),
                    "food_class": result.get("foodClass"),
                    "food_category": result.get("foodCategory"),
                    "gtin_upc": result.get("gtinUpc"),
                    "ingredients": result.get("ingredients"),
                    "serving_size": result.get("servingSize"),
                    "serving_size_unit": result.get("servingSizeUnit"),
                    "household_serving_fulltext": result.get(
                        "householdServingFullText"
                    ),
                    "modified_date": result.get("modifiedDate"),
                    "available_date": result.get("availableDate"),
                    "market_country": result.get("marketCountry"),
                    "discontinued_date": result.get("discontinuedDate"),
                    "preparation_state_code": result.get("preparationStateCode"),
                    "trade_channel": result.get("tradeChannel"),
                    "short_description": result.get("shortDescription"),
                    "nutrition_enhanced": result.get("nutrition_enhanced", {}),
                    "food_nutrients": result.get("foodNutrients", []),
                    "food_attributes": result.get("foodAttributes", []),
                    "food_attribute_types": result.get("foodAttributeTypes", []),
                    "food_version_ids": result.get("foodVersionIds", []),
                }
            )

        client.close()

        return {
            "query": query,
            "results_found": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        logger.error(f"Error searching nutrition data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search nutrition data: {str(e)}"
        )


@nutrition_setup.get("/database_stats/")
async def get_database_stats():
    """Get current database statistics"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]

        collections = db.list_collection_names()
        stats = {"collections": {}}

        for collection_name in collections:
            try:
                collection = db[collection_name]
                count = collection.count_documents({})

                # Get a sample document to show structure
                sample = collection.find_one()
                fields = list(sample.keys()) if sample else []

                stats["collections"][collection_name] = {
                    "document_count": count,
                    "sample_fields": fields[:10],  # First 10 fields
                    "total_fields": len(fields),
                }
            except Exception as e:
                stats["collections"][collection_name] = {"error": str(e)}

        # Get database stats
        db_stats = db.command("dbstats")

        client.close()

        return {
            "database_name": os.getenv("MONGO_DB_NAME", "usda_nutrition"),
            "total_collections": len(collections),
            "collection_details": stats["collections"],
            "database_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
            "storage_size_mb": round(db_stats.get("storageSize", 0) / (1024 * 1024), 2),
            "indexes": db_stats.get("indexes", 0),
        }

    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@nutrition_setup.post("/import_usda_data/")
async def import_usda_data(
    file_path: str = "./fast_api/app/api/nutrition_data/extracted/FoodData_Central_branded_food_json_2025-04-24.json",
):
    """Import USDA Branded Foods data into MongoDB with enhanced nutrition calculations"""

    def convert_decimal_in_dict(d):
        """Recursively convert all Decimal values to float in a dictionary"""
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = float(v)
            elif isinstance(v, dict):
                convert_decimal_in_dict(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        convert_decimal_in_dict(item)
        return d

    def extract_nutrient_by_id(food_nutrients, nutrient_id):
        """Extract specific nutrient amount by ID"""
        for nutrient in food_nutrients:
            if nutrient.get("nutrient", {}).get("id") == nutrient_id:
                return nutrient.get("amount", 0)
        return 0

    def calculate_per_100g_values(food_item):
        """Calculate per-100g nutrition values and add enhanced structure"""
        serving_size = food_item.get("servingSize", 100)

        if not serving_size or serving_size <= 0:
            serving_size = 100

        multiplier = 100 / serving_size
        food_nutrients = food_item.get("foodNutrients", [])

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

        label_nutrients = food_item.get("labelNutrients", {})
        enhanced_label = {}

        label_mapping = {
            "calories": "calories",
            "fat": "fat_g",
            "saturatedFat": "saturated_fat_g",
            "transFat": "trans_fat_g",
            "cholesterol": "cholesterol_mg",
            "sodium": "sodium_mg",
            "carbohydrates": "carbs_g",
            "fiber": "fiber_g",
            "sugars": "sugars_g",
            "protein": "protein_g",
        }

        for label_key, standard_key in label_mapping.items():
            if label_key in label_nutrients and "value" in label_nutrients[label_key]:
                enhanced_label[standard_key] = label_nutrients[label_key]["value"]

        food_item["nutrition_enhanced"] = {
            "serving_info": {
                "serving_size_g": serving_size,
                "serving_description": food_item.get("householdServingFullText", ""),
                "multiplier_to_100g": round(multiplier, 4),
            },
            "per_serving": per_serving,
            "per_100g": per_100g,
            "label_nutrients_enhanced": enhanced_label,
            "nutrition_density_score": calculate_nutrition_density_score(per_100g),
            "macro_breakdown": calculate_macro_breakdown(per_100g),
        }

        return food_item

    def calculate_nutrition_density_score(per_100g):
        """Calculate a simple nutrition density score"""
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
            logger.warning(f"Error calculating macro breakdown: {str(e)}")
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

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Get MongoDB connection
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        branded_foods = db["branded_foods"]

        logger.info("Creating enhanced indexes...")
        # Create indexes with error handling to avoid conflicts
        try:
            # Basic indexes
            branded_foods.create_index([("foodClass", pymongo.ASCENDING)])
            branded_foods.create_index([("brandOwner", pymongo.ASCENDING)])
            branded_foods.create_index([("foodCategory", pymongo.ASCENDING)])
            branded_foods.create_index([("gtinUpc", pymongo.ASCENDING)])

            # Create compound text index for search
            branded_foods.create_index(
                [("description", pymongo.TEXT), ("ingredients", pymongo.TEXT)],
                name="search_text_index",
            )

            # Nutrition indexes
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.primary_macro_category",
                        pymongo.ASCENDING,
                    )
                ]
            )
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.is_high_protein",
                        pymongo.ASCENDING,
                    )
                ]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_high_fat", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_high_carb", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_balanced", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.per_100g.protein_g", pymongo.DESCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.per_100g.energy_kcal", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.nutrition_density_score", pymongo.DESCENDING)]
            )
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.protein_percent",
                        pymongo.DESCENDING,
                    )
                ]
            )

            logger.info("All indexes created successfully")

        except Exception as e:
            logger.warning(
                f"Some indexes may already exist or failed to create: {str(e)}"
            )
            # Continue with import even if some indexes fail

        logger.info("Processing and inserting enhanced food data...")
        batch = []
        batch_size = 1000
        total_processed = 0
        enhanced_count = 0

        # Use ijson to stream the JSON
        with open(file_path, "rb") as f:
            parser = ijson.items(f, "BrandedFoods.item")

            for food in parser:
                try:
                    # Convert all Decimal values to float recursively
                    food = convert_decimal_in_dict(food)

                    # Add enhanced nutrition calculations
                    food = calculate_per_100g_values(food)
                    enhanced_count += 1

                    batch.append(food)

                    if len(batch) >= batch_size:
                        try:
                            branded_foods.insert_many(batch)
                            total_processed += len(batch)
                            logger.info(
                                f"Processed {total_processed} foods... ({enhanced_count} enhanced)"
                            )
                        except Exception as e:
                            logger.error(f"Error in batch: {str(e)}")
                            raise HTTPException(
                                status_code=500, detail=f"Import failed: {str(e)}"
                            )
                        batch = []

                except Exception as e:
                    logger.warning(f"Error processing food item: {str(e)}")
                    # Still add the item without enhancement if calculation fails
                    batch.append(food)

            # Insert any remaining items
            if batch:
                try:
                    branded_foods.insert_many(batch)
                    total_processed += len(batch)
                    logger.info(
                        f"Final batch processed. Total: {total_processed} foods"
                    )
                except Exception as e:
                    logger.error(f"Error in final batch: {str(e)}")
                    raise HTTPException(
                        status_code=500, detail=f"Import failed: {str(e)}"
                    )

        final_count = branded_foods.count_documents({})
        enhanced_final_count = branded_foods.count_documents(
            {"nutrition_enhanced": {"$exists": True}}
        )
        client.close()

        logger.info("Enhanced data import completed successfully!")

        return {
            "status": "success",
            "message": "USDA data imported successfully with enhanced nutrition calculations",
            "total_documents_processed": total_processed,
            "enhanced_documents": enhanced_final_count,
            "final_document_count": final_count,
            "enhancement_rate": (
                f"{(enhanced_final_count/final_count)*100:.1f}%"
                if final_count > 0
                else "0%"
            ),
            "indexes_created": [
                "foodClass",
                "brandOwner",
                "foodCategory",
                "gtinUpc",
                "search_text_index (description + ingredients)",
                "nutrition_enhanced.macro_breakdown.primary_macro_category",
                "nutrition_enhanced.macro_breakdown.is_high_protein",
                "nutrition_enhanced.macro_breakdown.is_high_fat",
                "nutrition_enhanced.macro_breakdown.is_high_carb",
                "nutrition_enhanced.macro_breakdown.is_balanced",
                "nutrition_enhanced.per_100g.protein_g (desc)",
                "nutrition_enhanced.per_100g.energy_kcal (asc)",
                "nutrition_enhanced.nutrition_density_score (desc)",
                "nutrition_enhanced.macro_breakdown.protein_percent (desc)",
            ],
            "sample_enhanced_structure": {
                "nutrition_enhanced": {
                    "serving_info": {
                        "serving_size_g": 15,
                        "serving_description": "1 PAN FRIED SLICE",
                        "multiplier_to_100g": 6.67,
                    },
                    "per_serving": {
                        "energy_kcal": 90,
                        "protein_g": 5,
                        "total_fat_g": 7,
                    },
                    "per_100g": {
                        "energy_kcal": 600,
                        "protein_g": 33.3,
                        "total_fat_g": 46.7,
                    },
                    "nutrition_density_score": 5.55,
                    "macro_breakdown": {
                        "protein_percent": 22.2,
                        "fat_percent": 70.0,
                        "carbs_percent": 0.0,
                        "calories_from_protein": 133.2,
                        "calories_from_fat": 420.3,
                        "calories_from_carbs": 0.0,
                        "primary_macro_category": "high_fat",
                        "macro_categories": ["high_fat"],
                        "is_high_protein": False,
                        "is_high_fat": True,
                        "is_high_carb": False,
                        "is_balanced": False,
                    },
                }
            },
        }

    except Exception as e:
        logger.error(f"Error importing USDA data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to import USDA data: {str(e)}"
        )


@nutrition_setup.post("/sample_usda_data/")
async def sample_usda_data(
    file_path: str = "./fast_api/app/api/nutrition_data/extracted/FoodData_Central_branded_food_json_2025-04-24.json",
    sample_size: int = 5000,
):
    """Sample USDA Branded Foods data evenly across macro categories from MongoDB and extract from USDA JSON"""

    def convert_decimal_in_dict(d):
        """Recursively convert all Decimal values to float in a dictionary"""
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = float(v)
            elif isinstance(v, dict):
                convert_decimal_in_dict(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        convert_decimal_in_dict(item)
        return d

    def extract_nutrient_by_id(food_nutrients, nutrient_id):
        """Extract specific nutrient amount by ID"""
        for nutrient in food_nutrients:
            if nutrient.get("nutrient", {}).get("id") == nutrient_id:
                return nutrient.get("amount", 0)
        return 0

    def calculate_per_100g_values(food_item):
        """Calculate per-100g nutrition values and add enhanced structure"""
        serving_size = food_item.get("servingSize", 100)

        if not serving_size or serving_size <= 0:
            serving_size = 100

        multiplier = 100 / serving_size
        food_nutrients = food_item.get("foodNutrients", [])

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

        label_nutrients = food_item.get("labelNutrients", {})
        enhanced_label = {}

        label_mapping = {
            "calories": "calories",
            "fat": "fat_g",
            "saturatedFat": "saturated_fat_g",
            "transFat": "trans_fat_g",
            "cholesterol": "cholesterol_mg",
            "sodium": "sodium_mg",
            "carbohydrates": "carbs_g",
            "fiber": "fiber_g",
            "sugars": "sugars_g",
            "protein": "protein_g",
        }

        for label_key, standard_key in label_mapping.items():
            if label_key in label_nutrients and "value" in label_nutrients[label_key]:
                enhanced_label[standard_key] = label_nutrients[label_key]["value"]

        food_item["nutrition_enhanced"] = {
            "serving_info": {
                "serving_size_g": serving_size,
                "serving_description": food_item.get("householdServingFullText", ""),
                "multiplier_to_100g": round(multiplier, 4),
            },
            "per_serving": per_serving,
            "per_100g": per_100g,
            "label_nutrients_enhanced": enhanced_label,
            "nutrition_density_score": calculate_nutrition_density_score(per_100g),
            "macro_breakdown": calculate_macro_breakdown(per_100g),
        }

        return food_item

    def calculate_nutrition_density_score(per_100g):
        """Calculate a simple nutrition density score"""
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
            logger.warning(f"Error calculating macro breakdown: {str(e)}")
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

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        logger.info(
            f"Starting to sample {sample_size} foods evenly across macro categories from MongoDB..."
        )

        # Connect to MongoDB and get nutrition_enhanced foods by macro category
        client = get_mongo_client()
        db_name = os.getenv("MONGO_DB_NAME", "usda_nutrition")
        db = client[db_name]
        collection = db["branded_foods"]

        # Define macro categories for even sampling
        macro_categories = ["high_protein", "high_fat", "high_carb", "balanced"]
        foods_per_category = sample_size // len(
            macro_categories
        )  # 1250 foods per category
        remainder = sample_size % len(macro_categories)

        logger.info(
            f"Sampling {foods_per_category} foods per macro category ({len(macro_categories)} categories)"
        )

        # Sample evenly from each macro category
        all_sampled_fdc_ids = []
        category_counts = {}

        import random

        random.seed(42)  # For reproducible sampling

        for i, category in enumerate(macro_categories):
            # Calculate sample size for this category (add remainder to first categories)
            category_sample_size = foods_per_category + (1 if i < remainder else 0)

            # Query MongoDB for foods in this macro category with nutrition_enhanced data
            query = {
                "nutrition_enhanced.macro_breakdown.primary_macro_category": category,
                "nutrition_enhanced": {"$exists": True},
                "fdcId": {"$exists": True},
            }

            # Get all foods in this category and sample from them
            foods_in_category = list(collection.find(query, {"fdcId": 1, "_id": 0}))

            if len(foods_in_category) == 0:
                logger.warning(f"No foods found for macro category: {category}")
                continue

            # Sample from available foods in this category
            if len(foods_in_category) <= category_sample_size:
                sampled_foods = foods_in_category
            else:
                sampled_foods = random.sample(foods_in_category, category_sample_size)

            # Extract fdcIds
            fdc_ids = [food["fdcId"] for food in sampled_foods]
            all_sampled_fdc_ids.extend(fdc_ids)
            category_counts[category] = len(fdc_ids)

            logger.info(
                f"Sampled {len(fdc_ids)} foods from {category} category (available: {len(foods_in_category)})"
            )

        client.close()

        logger.info(f"Total fdcIds sampled: {len(all_sampled_fdc_ids)}")
        logger.info(f"Category distribution: {category_counts}")

        # Now extract these specific foods from the USDA JSON file
        logger.info("Extracting sampled foods from USDA JSON file...")
        sampled_foods = []
        fdc_id_set = set(
            str(fdc_id) for fdc_id in all_sampled_fdc_ids
        )  # Convert to strings for comparison

        # Use ijson to stream and extract specific foods
        with open(file_path, "rb") as f:
            parser = ijson.items(f, "BrandedFoods.item")

            for food in parser:
                food_fdc_id = str(food.get("fdcId", ""))
                if food_fdc_id in fdc_id_set:
                    sampled_foods.append(food)
                    fdc_id_set.remove(
                        food_fdc_id
                    )  # Remove to avoid duplicates and speed up

                    # Stop when we've found all foods
                    if not fdc_id_set:
                        break

        logger.info(f"Successfully extracted {len(sampled_foods)} foods from USDA JSON")

        # Process sampled foods with enhanced nutrition calculations
        logger.info("Processing sampled foods with enhanced nutrition calculations...")
        processed_foods = []

        for food in sampled_foods:
            try:
                # Convert all Decimal values to float recursively
                food = convert_decimal_in_dict(food)

                # Add enhanced nutrition calculations
                food = calculate_per_100g_values(food)
                processed_foods.append(food)

            except Exception as e:
                logger.warning(f"Error processing food item: {str(e)}")
                # Still add the item without enhancement if calculation fails
                processed_foods.append(food)

        # Create output directory if it doesn't exist
        output_dir = Path("./fast_api/app/api/nutrition_data/samples")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sampled data to file
        output_file = output_dir / f"usda_sampled_{sample_size}_foods.json"

        sample_data = {
            "metadata": {
                "sampled_foods": len(processed_foods),
                "sample_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": file_path,
                "random_seed": 42,
                "sampling_method": "even_distribution_across_macro_categories",
                "macro_categories": macro_categories,
                "category_distribution": category_counts,
                "foods_per_category_target": foods_per_category,
            },
            "BrandedFoods": processed_foods,
        }

        with open(output_file, "w") as f:
            json.dump(sample_data, f, indent=2)

        logger.info(f"Sample data saved to {output_file}")

        return {
            "status": "success",
            "message": f"Successfully sampled {len(processed_foods)} foods evenly across macro categories",
            "sampled_food_count": len(processed_foods),
            "category_distribution": category_counts,
            "output_file": str(output_file),
            "file_size_mb": round(os.path.getsize(output_file) / (1024 * 1024), 2),
            "sample_metadata": sample_data["metadata"],
            "sample_food_example": processed_foods[0] if processed_foods else None,
        }

    except Exception as e:
        logger.error(f"Error sampling USDA data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to sample USDA data: {str(e)}"
        )


@nutrition_setup.post("/import_sampled_data/")
async def import_sampled_data(
    sample_file: str = "./fast_api/app/api/nutrition_data/samples/usda_sampled_5000_foods.json",
):
    """Import sampled USDA data into MongoDB 'branded_foods_sample' collection for quick setup"""

    try:
        # Check if file exists
        if not os.path.exists(sample_file):
            raise HTTPException(
                status_code=404, detail=f"Sample file not found: {sample_file}"
            )

        # Get MongoDB connection
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        branded_foods = db["branded_foods_sample"]

        logger.info("Creating enhanced indexes...")
        # Create indexes with error handling to avoid conflicts
        try:
            # Basic indexes
            branded_foods.create_index([("foodClass", pymongo.ASCENDING)])
            branded_foods.create_index([("brandOwner", pymongo.ASCENDING)])
            branded_foods.create_index([("foodCategory", pymongo.ASCENDING)])
            branded_foods.create_index([("gtinUpc", pymongo.ASCENDING)])

            # Create compound text index for search
            branded_foods.create_index(
                [("description", pymongo.TEXT), ("ingredients", pymongo.TEXT)],
                name="search_text_index",
            )

            # Nutrition indexes
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.primary_macro_category",
                        pymongo.ASCENDING,
                    )
                ]
            )
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.is_high_protein",
                        pymongo.ASCENDING,
                    )
                ]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_high_fat", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_high_carb", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.macro_breakdown.is_balanced", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.per_100g.protein_g", pymongo.DESCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.per_100g.energy_kcal", pymongo.ASCENDING)]
            )
            branded_foods.create_index(
                [("nutrition_enhanced.nutrition_density_score", pymongo.DESCENDING)]
            )
            branded_foods.create_index(
                [
                    (
                        "nutrition_enhanced.macro_breakdown.protein_percent",
                        pymongo.DESCENDING,
                    )
                ]
            )

            logger.info("All indexes created successfully")

        except Exception as e:
            logger.warning(
                f"Some indexes may already exist or failed to create: {str(e)}"
            )
            # Continue with import even if some indexes fail

        # Load and import sampled data
        logger.info(f"Loading sampled data from {sample_file}...")

        with open(sample_file, "r") as f:
            sample_data = json.load(f)

        foods = sample_data.get("BrandedFoods", [])
        metadata = sample_data.get("metadata", {})

        logger.info(f"Importing {len(foods)} sampled foods...")

        batch_size = 1000
        total_processed = 0

        for i in range(0, len(foods), batch_size):
            batch = foods[i : i + batch_size]
            try:
                branded_foods.insert_many(batch)
                total_processed += len(batch)
                logger.info(f"Processed {total_processed}/{len(foods)} foods...")
            except Exception as e:
                logger.error(f"Error in batch: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

        final_count = branded_foods.count_documents({})
        enhanced_final_count = branded_foods.count_documents(
            {"nutrition_enhanced": {"$exists": True}}
        )
        client.close()

        logger.info("Sampled data import completed successfully!")

        return {
            "status": "success",
            "message": "Sampled USDA data imported successfully",
            "sample_metadata": metadata,
            "total_documents_imported": total_processed,
            "enhanced_documents": enhanced_final_count,
            "final_document_count": final_count,
            "source_file": sample_file,
            "indexes_created": [
                "foodClass",
                "brandOwner",
                "foodCategory",
                "gtinUpc",
                "search_text_index (description + ingredients)",
                "nutrition_enhanced.macro_breakdown.primary_macro_category",
                "nutrition_enhanced.macro_breakdown.is_high_protein",
                "nutrition_enhanced.macro_breakdown.is_high_fat",
                "nutrition_enhanced.macro_breakdown.is_high_carb",
                "nutrition_enhanced.macro_breakdown.is_balanced",
                "nutrition_enhanced.per_100g.protein_g (desc)",
                "nutrition_enhanced.per_100g.energy_kcal (asc)",
                "nutrition_enhanced.nutrition_density_score (desc)",
                "nutrition_enhanced.macro_breakdown.protein_percent (desc)",
            ],
        }

    except Exception as e:
        logger.error(f"Error importing sampled data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to import sampled data: {str(e)}"
        )


@nutrition_setup.get("/database_availability/")
async def check_database_availability():
    """Check which databases are available (full vs sample)"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        
        # Check if full database exists and has data
        full_collection = db["branded_foods"]
        full_count = full_collection.count_documents({})
        full_available = full_count > 0
        
        # Check if sample database exists and has data
        sample_collection = db["branded_foods_sample"]
        sample_count = sample_collection.count_documents({})
        sample_available = sample_count > 0
        
        client.close()
        
        return {
            "full_database": {
                "available": full_available,
                "document_count": full_count,
                "collection_name": "branded_foods"
            },
            "sample_database": {
                "available": sample_available,
                "document_count": sample_count,
                "collection_name": "branded_foods_sample"
            },
            "recommendation": "full" if full_available else "sample" if sample_available else "none"
        }
        
    except Exception as e:
        logger.error(f"Error checking database availability: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check database availability: {str(e)}"
        )
