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


"""
version: '3.3'

services:
  # Jupyter Notebook for AI/ML development
  notebook_ai_fitness_planner:
    restart: always
    build: ./jupyter_notebook
    ports:
      - "1960:3839"
    volumes:
      - ./:/app
    env_file:
      - .env
    environment:
      - JUPYTER_TOKEN=${JUPYTER_TOKEN}
      - JUPYTER_PASSWORD_HASH=${JUPYTER_PASSWORD_HASH}
      - MONGODB_URI=mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb_ai_fitness_planner:27017/
    command: jupyter lab --port=3839 --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.password="${JUPYTER_PASSWORD_HASH}"
    networks:
      - ai-fitness-network
    depends_on:
      - ai_fitness_planner_db
      - mongodb_ai_fitness_planner
    
  # PostgreSQL Database
  ai_fitness_planner_db:
    image: postgres:15-alpine
    container_name: ai_fitness_planner_db
    restart: always
    env_file:
      - .env
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    ports:
      - "4553:5432"
    volumes:
      - db_data_ai_fitness_planner:/var/lib/postgresql/data
    networks:
      - ai-fitness-network
      - sp-net
  
# MongoDB Database for USDA Nutrition Data
  mongodb_ai_fitness_planner:
    image: mongo:7.0
    container_name: mongodb_ai_fitness_planner
    restart: always
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_USER}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
      MONGO_INITDB_DATABASE: ${MONGO_DB_NAME}
    ports:
      - "27019:27017"  # Different port to avoid conflicts
    volumes:
      - mongodb_data_ai_fitness_planner:/data/db
      - ./data/mongodb_init:/docker-entrypoint-initdb.d  # For init scripts
    networks:
      - ai-fitness-network
      - sp-net

  # Mongo Express for MongoDB management
  mongo_express_ai_fitness_planner:
    image: mongo-express:1.0.0
    container_name: mongo_express_ai_fitness_planner
    restart: always
    ports:
      - "8084:8081"  # Different port to avoid conflicts
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: ${MONGO_USER}
      ME_CONFIG_MONGODB_ADMINPASSWORD: ${MONGO_PASSWORD}
      ME_CONFIG_MONGODB_URL: mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb_ai_fitness_planner:27017/
      ME_CONFIG_BASICAUTH_USERNAME: ${MONGO_EXPRESS_USER}
      ME_CONFIG_BASICAUTH_PASSWORD: ${MONGO_EXPRESS_PASSWORD}
    networks:
      - ai-fitness-network
      - sp-net
    depends_on:
      - mongodb_ai_fitness_planner

  # pgAdmin for database management
  pgadmin_ai_fitness_planner:
    container_name: pgadmin4_ai_fitness_planner
    image: dpage/pgadmin4:latest
    restart: always
    volumes:
      - pgadmin_ai_fitness_planner:/var/lib/pgadmin
    env_file:
      - .env
    environment:
      - PGADMIN_DEFAULT_EMAIL=${PGADMIN_EMAIL}
      - PGADMIN_DEFAULT_PASSWORD=${PGADMIN_PASSWORD}
      - PGADMIN_CONFIG_SERVER_MODE=False
    ports:
      - "4053:80"
    networks:
      - ai-fitness-network
      - sp-net
    depends_on:
      - ai_fitness_planner_db

  # FastAPI Backend
  fast_api_ai_fitness_planner:
    restart: always
    build: 
      context: ./fast_api
      dockerfile: Dockerfile-dev
    env_file:
      - .env
    volumes:
      - ./:/app
    ports:
      - "1015:8000"
    command: ["--host", "0.0.0.0", "fast_api.app.main:app", "--reload"]
    networks:  # ADD THIS SECTION
      - ai-fitness-network
      - sp-net
    depends_on:
      - ai_fitness_planner_db
      - mongodb_ai_fitness_planner

  # Streamlit Frontend
  streamlit_app_ai_fitness_planner:
    build: 
      context: ./streamlit
      dockerfile: Dockerfile
    restart: always
    env_file:
      - .env
    command: "streamlit run streamlit/🏠_home.py"
    ports:
      - "8526:8501"
    volumes:
      - ./streamlit:/usr/src/app
    networks:
      - ai-fitness-network
      - sp-net
    depends_on:
      - ai_fitness_planner_db
      - mongodb_ai_fitness_planner

# Named volumes for data persistence
volumes:
  db_data_ai_fitness_planner:
    driver: local
  pgadmin_ai_fitness_planner:
    driver: local
  mongodb_data_ai_fitness_planner:
    driver: local

# Networks
networks:
  ai-fitness-network:
    driver: bridge
  sp-net:
    external: true
"""


@workout.get("/load_usda_data/")
async def load_usda_data():
    """Load USDA Branded Foods data into PostgreSQL database"""
    import psycopg2
    from psycopg2 import sql

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


# Add an endpoint to check if USDA data exists
@workout.get("/check_usda_data/")
async def check_usda_data():
    """Check if USDA nutrition data exists in MongoDB"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]

        # Check for branded foods collection
        collections = db.list_collection_names()

        if "branded_foods" in collections:
            count = db.branded_foods.count_documents({})

            # Get a sample document
            sample = db.branded_foods.find_one()

            client.close()

            return {
                "status": "data_exists",
                "collection": "branded_foods",
                "document_count": count,
                "sample_fields": list(sample.keys()) if sample else [],
                "message": f"Found {count:,} branded food documents",
            }
        else:
            client.close()
            return {
                "status": "no_data",
                "available_collections": collections,
                "message": "No branded_foods collection found. You may need to import USDA data.",
            }

    except Exception as e:
        logger.error(f"Error checking USDA data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check USDA data: {str(e)}"
        )


# Simple nutrition search endpoint to test functionality
@workout.get("/search_nutrition/")
async def search_nutrition(query: str = "coca cola", limit: int = 5):
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
                    "description": result.get("description", ""),
                    "brand": result.get("brandOwner", ""),
                    "category": result.get("brandedFoodCategory", ""),
                    "fdcId": result.get("fdcId", ""),
                    "calories": result.get("labelNutrients", {})
                    .get("calories", {})
                    .get("value", 0),
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
