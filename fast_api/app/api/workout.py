import os
import json
import logging
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
import re

import requests
import zipfile
import os
from pathlib import Path
import time
from tqdm import tqdm
import json
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

workout = APIRouter()


class Exercise(BaseModel):
    """Individual exercise details"""

    name: str = Field(description="Name of the exercise")
    description: str = Field(description="Exercise instructions and key points")
    difficulty: str = Field(description="Difficulty level of the exercise")
    target_muscles: str = Field(description="Primary muscle groups targeted")
    duration: int = Field(description="Duration in minutes", default=1)
    sets: int = Field(description="Number of sets", default=3)
    work_time: str = Field(description="Work time per set (e.g., '40 seconds')")
    rest_time: str = Field(description="Rest time between sets (e.g., '20 seconds')")
    equipment_needed: str = Field(description="Equipment required for this exercise")


class WorkoutPlan(BaseModel):
    """Complete workout plan"""

    exercises: List[Exercise] = Field(
        description="List of exercises", min_items=5, max_items=20
    )
    total_duration: int = Field(description="Total workout duration in minutes")
    skill_level_match: bool = Field(description="Whether exercises match skill level")
    equipment_appropriate: bool = Field(
        description="Whether exercises match available equipment"
    )
    workout_type: str = Field(
        description="Type of workout (strength, cardio, mixed, etc.)"
    )


class AudioScriptRequest(BaseModel):
    """Request for generating workout audio script"""

    workout_plan: Dict[str, Any] = Field(description="Complete workout plan")
    trainer_voice_style: str = Field(
        description="Trainer voice style (Motivational, Calm, etc.)"
    )


class AudioScriptResponse(BaseModel):
    """Response containing the generated audio script"""

    audio_script: str = Field(description="Complete audio script for the workout")


@workout.post("/generate_workout/")
async def generate_workout(query: Dict[str, Any]) -> Response:
    """Generate personalized workout plan based on available equipment and target muscle groups"""
    try:
        skill_level = query.get("skill_level", "beginner")
        workout_length = query.get("workout_length", 30)
        explanation_detail = query.get("explanation_detail", "basic")
        equipment_available = query.get("equipment_available", ["bodyweight"])
        muscle_groups = query.get("muscle_groups", ["full body"])
        rest_time = query.get("rest_time", 30)

        # Calculate work time (each set is 1 minute total)
        work_time = 60 - rest_time

        # Convert equipment list to readable string
        if "bodyweight" in equipment_available and len(equipment_available) == 1:
            equipment_str = "bodyweight exercises only (no equipment needed)"
        else:
            # Filter out bodyweight if other equipment is available
            equipment_list = [eq for eq in equipment_available if eq != "bodyweight"]
            if equipment_list:
                equipment_str = ", ".join(equipment_list)
                if "bodyweight" in equipment_available:
                    equipment_str += " (bodyweight exercises can also be included)"
            else:
                equipment_str = "bodyweight exercises only (no equipment needed)"

        # Convert muscle groups list to readable string
        muscle_groups_str = ", ".join(muscle_groups)

        # Special handling for full body
        if "full body" in muscle_groups:
            if len(muscle_groups) == 1:
                muscle_focus = "a full body workout targeting all major muscle groups"
            else:
                other_groups = [mg for mg in muscle_groups if mg != "full body"]
                muscle_focus = f"a full body workout with extra emphasis on: {', '.join(other_groups)}"
        else:
            muscle_focus = f"the following muscle groups: {muscle_groups_str}"

        OPEN_AI_MODEL = "gpt-4o"
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=OPEN_AI_MODEL)

        prompt = f"""Create a {workout_length}-minute workout plan for {skill_level} level fitness.
        Available equipment: {equipment_str}
        Target muscle groups: Focus on {muscle_focus}
        
        CRITICAL TIMING STRUCTURE:
        - Each set is exactly 1 minute total: {work_time} seconds of work + {rest_time} seconds of rest
        - Work time: {work_time} seconds of continuous, challenging exercise
        - Rest time: {rest_time} seconds between sets
        - For weighted exercises, emphasize choosing challenging weights that can be maintained for the full {work_time} seconds
        - Reps are not the focus - time under tension is the key metric
        
        Provide {explanation_detail} explanations for each exercise.
        
        Requirements:
        - Only use exercises that can be performed with the available equipment
        - Prioritize exercises that target the specified muscle groups: {muscle_groups_str}
        - If "full body" is selected, ensure the workout includes exercises for all major muscle groups
        - If specific muscle groups are selected, ensure at least 70% of exercises target those areas
        - Ensure exercises are appropriate for the {skill_level} skill level
        - Each exercise should specify sets and the timing structure ({work_time}s work + {rest_time}s rest)
        - Progress exercises logically from warm-up to cool-down
        - Include variety in movement patterns and exercise types
        - For weighted exercises, emphasize weight selection for time-based work rather than rep counting
        
        Equipment clarifications:
        - "bodyweight" = exercises using only body weight (push-ups, squats, etc.)
        - "dumbbells" = adjustable dumbbells or dumbbell set
        - "barbell" = barbell with weight plates
        - "resistance bands" = elastic resistance bands or tubes
        
        Format each exercise with:
        - Name of the exercise
        - Clear description with proper form cues, safety tips, and weight selection guidance
        - Difficulty level (beginner/intermediate/advanced)
        - Target muscle groups (be specific about primary and secondary muscles)
        - Equipment needed (must match available equipment)
        - Number of sets
        - Work time: "{work_time} seconds" (time under tension/continuous work)
        - Rest time: "{rest_time} seconds" (recovery between sets)
        
        IMPORTANT WEIGHT SELECTION GUIDANCE:
        - For weighted exercises, include guidance on selecting challenging weights
        - Emphasize that weights should be challenging enough to maintain good form for the full {work_time} seconds
        - Mention that the goal is time under tension, not maximum reps
        - For bodyweight exercises, suggest modifications to adjust difficulty for the time period
        
        Make sure the total workout duration matches the requested {workout_length} minutes."""

        structured_llm = llm.with_structured_output(WorkoutPlan)
        response = structured_llm.invoke([HumanMessage(content=prompt)])

        return Response(
            content=json.dumps(response.dict(), indent=2), media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Error generating workout plan: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate workout plan: {str(e)}"
        )


@workout.post("/generate_audio_script/")
async def generate_audio_script(request: AudioScriptRequest) -> AudioScriptResponse:
    """Generate audio script for workout with timed instructions and trainer personality"""
    try:
        workout_plan = request.workout_plan
        trainer_style = request.trainer_voice_style

        # Extract timing information from first exercise to understand structure
        first_exercise = workout_plan["exercises"][0]
        work_time_str = first_exercise["work_time"]
        rest_time_str = first_exercise["rest_time"]

        # Extract numbers from time strings
        work_seconds = int(re.search(r"\d+", work_time_str).group())
        rest_seconds = int(re.search(r"\d+", rest_time_str).group())

        # Define trainer personality based on style
        trainer_personalities = {
            "Motivational": "energetic, encouraging, and enthusiastic. Use phrases like 'You've got this!', 'Push through!', 'Great job!', and 'Stay strong!'",
            "Calm & Encouraging": "supportive, gentle, and reassuring. Use phrases like 'Take your time', 'Focus on your breathing', 'You're doing great', and 'Stay centered'",
            "Professional": "clear, instructional, and focused. Use precise language, proper form cues, and technical guidance",
            "Energetic": "high-energy, dynamic, and exciting. Use phrases like 'Let's go!', 'Power through!', 'Feel the burn!', and 'Amazing work!'",
        }

        trainer_personality = trainer_personalities.get(
            trainer_style, trainer_personalities["Motivational"]
        )

        OPEN_AI_MODEL = "gpt-4o"
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=OPEN_AI_MODEL)

        prompt = f"""Create a complete audio script for a workout trainer guiding someone through this workout plan:

WORKOUT PLAN:
{json.dumps(workout_plan, indent=2)}

TRAINER STYLE: {trainer_style}
Your trainer personality should be: {trainer_personality}

TIMING STRUCTURE:
- Work time: {work_seconds} seconds per set
- Rest time: {rest_seconds} seconds between sets
- Each set = {work_seconds + rest_seconds} seconds total

AUDIO SCRIPT REQUIREMENTS:

1. **Introduction (30-45 seconds)**:
   - Welcome message in {trainer_style.lower()} tone
   - Brief overview of the workout
   - Motivation and preparation cues
   - "Let's get started!" transition

2. **For Each Exercise**:
   - Exercise name announcement
   - Quick form reminder (15-20 seconds)
   - Set-by-set guidance with timing:
     
   **For each set:**
   - "Set [number] - Get ready!"
   - "3, 2, 1, GO!" (start countdown)
   - Mid-exercise motivation and form cues during work time
   - "Time! Great job! Rest now." (at end of work time)
   - Rest period encouragement and next set preparation
   - Repeat for all sets

3. **Exercise Transitions**:
   - "Next up: [exercise name]"
   - Equipment setup if needed
   - Brief form reminder

4. **Workout Conclusion (30-45 seconds)**:
   - Congratulations and accomplishment acknowledgment
   - Cool-down reminder
   - Motivational closing message

SPECIFIC TIMING GUIDANCE:
- Give 10-second countdowns for exercise starts
- Provide encouragement every 10-15 seconds during work periods
- Use the exact timing: {work_seconds} seconds work, {rest_seconds} seconds rest
- Include transition time between exercises (15-20 seconds)

TONE AND LANGUAGE:
- Match the {trainer_style} personality throughout
- Use second person ("you") to directly address the user
- Include specific form cues and safety reminders
- Vary motivational phrases to avoid repetition
- Keep energy appropriate to the trainer style chosen

The script should be written as if spoken aloud, with natural pauses and breathing indicated by periods and commas.
Include timing cues in brackets like [10 seconds remaining] to help with audio timing.
Make it engaging and personalized to keep the user motivated throughout the entire workout.

Total estimated audio length should match the workout duration: {workout_plan['total_duration']} minutes."""

        response = llm.invoke([HumanMessage(content=prompt)])

        return AudioScriptResponse(audio_script=response.content)

    except Exception as e:
        logger.error(f"Error generating audio script: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate audio script: {str(e)}"
        )


@workout.get("/test_postgres_db/")
async def test_db():
    """Test database connection with improved error handling"""
    import psycopg2
    from psycopg2 import sql

    # Try different host configurations
    hosts_to_try = [
        "ai_fitness_planner_db",  # Docker service name
        "localhost",  # If running locally
        "127.0.0.1",  # Fallback
    ]

    for host in hosts_to_try:
        try:
            logger.info(f"Attempting to connect to database at host: {host}")

            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=host,
                port=5432,
                connect_timeout=10,
            )

            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()
            cursor.close()
            conn.close()

            logger.info(f"Successfully connected to database at host: {host}")
            return {
                "status": "success",
                "db_version": db_version,
                "connected_host": host,
                "environment_vars": {
                    "DB_NAME": os.getenv("DB_NAME"),
                    "DB_USER": os.getenv("DB_USER"),
                    "DB_PASSWORD": "***" if os.getenv("DB_PASSWORD") else None,
                },
            }

        except Exception as e:
            logger.warning(f"Failed to connect to database at host {host}: {str(e)}")
            continue

    # If all hosts failed
    error_msg = f"Failed to connect to database. Tried hosts: {', '.join(hosts_to_try)}"
    logger.error(error_msg)
    raise HTTPException(status_code=500, detail=error_msg)


@workout.get("/download_raw_usda_data/")
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
@workout.get("/test_mongo_db/")
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

@workout.get("/search_nutrition/")
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


@workout.get("/database_stats/")
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


@workout.post("/import_usda_data/")
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
