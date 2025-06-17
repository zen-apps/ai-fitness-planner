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

# Corrected imports for static file hosting
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

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


# connect to postgres database

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
    command: jupyter lab --port=3839 --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.password="${JUPYTER_PASSWORD_HASH}"
    networks:
      - ai-fitness-network
    depends_on:
      - ai_fitness_planner_db
    
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
    depends_on:
      - ai_fitness_planner_db

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

# Named volumes for data persistence
volumes:
  db_data_ai_fitness_planner:
    driver: local
  pgadmin_ai_fitness_planner:
    driver: local

# Networks
networks:
  ai-fitness-network:
    driver: bridge
  sp-net:
    external: true

    (.venv) jjanzen@zen-general-vm:~/localfiles/ai-fitness-planner$ ls -la
total 44
drwxr-xr-x  6 jjanzen jjanzen 4096 Jun 17 01:03 .
drwxr-xr-x 44 jjanzen jjanzen 4096 Jun 16 20:10 ..
-rw-r--r--  1 jjanzen jjanzen 1218 Jun 16 21:05 .env
drwxr-xr-x  9 jjanzen jjanzen 4096 Jun 17 01:04 .git
-rw-r--r--  1 jjanzen jjanzen  767 Jun 17 01:03 .gitignore
-rw-r--r--  1 jjanzen jjanzen    0 Jun 16 20:16 CLAUDE.md
-rw-r--r--  1 jjanzen jjanzen  395 Jun 16 20:36 Makefile
-rw-r--r--  1 jjanzen jjanzen   20 Jun 16 20:04 README.md
-rw-r--r--  1 jjanzen jjanzen 2559 Jun 17 00:51 docker-compose-dev.yml
drwxr-xr-x  3 jjanzen jjanzen 4096 Jun 17 16:39 fast_api
drwxr-xr-x 11 jjanzen jjanzen 4096 Jun 16 20:15 jupyter_notebook
drwxr-xr-x  4 jjanzen jjanzen 4096 Jun 16 20:16 streamlit
"""


@workout.get("test_db/")
async def test_db():
    """Test database connection"""
    import psycopg2
    from psycopg2 import sql

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host="ai_fitness_planner_db",
            port=5432,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        cursor.close()
        conn.close()
        return {"status": "success", "db_version": db_version}
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database connection failed: {str(e)}"
        )
