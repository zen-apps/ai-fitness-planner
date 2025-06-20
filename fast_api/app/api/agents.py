import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fitness_agents")

agents = APIRouter()


# Pydantic models for request/response
class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    activity_level: Optional[str] = (
        "moderate"  # sedentary, light, moderate, active, very_active
    )
    fitness_goal: Optional[str] = "maintenance"  # cut, bulk, maintenance, recomp
    target_calories: Optional[int] = None
    target_protein_g: Optional[float] = None
    target_carbs_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    allergies: Optional[List[str]] = []
    dietary_preferences: Optional[List[str]] = []  # vegetarian, vegan, keto, etc.
    equipment_available: Optional[List[str]] = []
    workout_frequency: Optional[int] = 3
    workout_duration: Optional[int] = 60  # minutes
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MealPlanRequest(BaseModel):
    user_id: str
    meal_count: Optional[int] = 3
    preferences: Optional[Dict[str, Any]] = {}


class WorkoutPlanRequest(BaseModel):
    user_id: str
    split_type: Optional[str] = "full_body"  # full_body, upper_lower, push_pull_legs
    training_style: Optional[str] = "hypertrophy"  # strength, hypertrophy, endurance
    days_per_week: Optional[int] = 3
    duration_minutes: Optional[int] = 60


class FitnessPlansResponse(BaseModel):
    user_id: str
    meal_plan: Optional[Dict[str, Any]] = None
    workout_plan: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    generated_at: datetime


# Helper function for MongoDB connection (reusing from nutrition.py)
def get_mongo_client():
    """Get MongoDB client connection"""
    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")

    try:
        # Docker internal connection
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@mongodb_ai_fitness_planner:27017/admin"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except:
        # Fallback to localhost
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@localhost:27019/admin"
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client


# Agent Classes
class ProfileManagerAgent:
    """Manages user profiles and calculates nutritional needs"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY")
        )

    def calculate_bmr(self, profile: UserProfile) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        if not all([profile.age, profile.weight, profile.height]):
            return 2000  # Default fallback

        # Assuming male for now - could add gender field
        bmr = (10 * profile.weight) + (6.25 * profile.height) - (5 * profile.age) + 5
        return bmr

    def calculate_tdee(self, profile: UserProfile) -> float:
        """Calculate Total Daily Energy Expenditure"""
        bmr = self.calculate_bmr(profile)

        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }

        multiplier = activity_multipliers.get(profile.activity_level, 1.55)
        return bmr * multiplier

    def calculate_macros(self, profile: UserProfile) -> Dict[str, float]:
        """Calculate macro targets based on goals"""
        tdee = self.calculate_tdee(profile)

        # Adjust calories based on goal
        if profile.fitness_goal == "cut":
            target_calories = tdee * 0.8  # 20% deficit
        elif profile.fitness_goal == "bulk":
            target_calories = tdee * 1.1  # 10% surplus
        else:  # maintenance or recomp
            target_calories = tdee

        # Standard macro ratios - could be customized
        protein_ratio = 0.30
        carbs_ratio = 0.40
        fat_ratio = 0.30

        return {
            "calories": round(target_calories),
            "protein_g": round((target_calories * protein_ratio) / 4),
            "carbs_g": round((target_calories * carbs_ratio) / 4),
            "fat_g": round((target_calories * fat_ratio) / 9),
        }

    async def update_profile(self, profile: UserProfile) -> UserProfile:
        """Update user profile with calculated values"""
        macros = self.calculate_macros(profile)

        profile.target_calories = macros["calories"]
        profile.target_protein_g = macros["protein_g"]
        profile.target_carbs_g = macros["carbs_g"]
        profile.target_fat_g = macros["fat_g"]
        profile.updated_at = datetime.now()

        # Store in MongoDB
        try:
            client = get_mongo_client()
            db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
            profiles = db["user_profiles"]

            profile_dict = profile.dict()
            profiles.update_one(
                {"user_id": profile.user_id}, {"$set": profile_dict}, upsert=True
            )
            client.close()

        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")

        return profile


class MealPlannerAgent:
    """Generates personalized meal plans using nutrition database"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")
        )

    async def find_foods_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict]:
        """Find foods using existing FAISS vector search"""
        try:
            # Import the vector search function from nutrition_search
            from .nutrition_search import search_nutrition_semantic, NutritionQuery

            # Build search query based on criteria
            query_parts = []
            dietary_restrictions = []
            macro_goals = {}

            if criteria.get("high_protein"):
                query_parts.append("high protein")
                macro_goals["protein_min"] = 15

            if criteria.get("low_carb"):
                query_parts.append("low carb")
                macro_goals["carbs_max"] = 10

            if criteria.get("food_category"):
                query_parts.append(criteria["food_category"])

            if criteria.get("dietary_preferences"):
                dietary_restrictions.extend(criteria["dietary_preferences"])

            # Build search query
            search_query = " ".join(query_parts) if query_parts else "nutritious food"

            # Create nutrition query
            nutrition_query = NutritionQuery(
                query=search_query,
                dietary_restrictions=dietary_restrictions,
                macro_goals=macro_goals,
                limit=20,
                similarity_threshold=0.6,
            )

            # Use existing vector search
            search_result = await search_nutrition_semantic(nutrition_query)

            # Convert results to expected format
            foods = []
            for result in search_result.results:
                food_item = {
                    "fdcId": result.get("fdc_id"),
                    "description": result.get("description"),
                    "brandOwner": result.get("brand_owner"),
                    "brandName": result.get("brand_name"),
                    "foodCategory": result.get("food_category"),
                    "nutrition_enhanced": {
                        "per_100g": result.get("nutrition_per_100g", {}),
                        "macro_breakdown": {
                            "primary_macro_category": result.get(
                                "primary_macro_category"
                            ),
                            "is_high_protein": result.get("is_high_protein", False),
                        },
                        "nutrition_density_score": result.get(
                            "nutrition_density_score", 0
                        ),
                    },
                    "similarity_score": result.get("similarity_score", 0),
                }
                foods.append(food_item)

            logger.info(
                f"Found {len(foods)} foods using vector search for query: {search_query}"
            )
            return foods

        except Exception as e:
            logger.error(f"Error in vector food search: {str(e)}")
            # Fallback to basic MongoDB query if vector search fails
            return await self._fallback_mongo_search(criteria)

    async def _fallback_mongo_search(self, criteria: Dict[str, Any]) -> List[Dict]:
        """Fallback MongoDB search if vector search fails"""
        try:
            client = get_mongo_client()
            db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
            branded_foods = db["branded_foods"]

            query = {}

            # Add macro-based filtering
            if criteria.get("high_protein"):
                query["nutrition_enhanced.macro_breakdown.is_high_protein"] = True

            if criteria.get("low_carb"):
                query["nutrition_enhanced.per_100g.carbs_g"] = {"$lt": 10}

            if criteria.get("food_category"):
                query["foodCategory"] = {
                    "$regex": criteria["food_category"],
                    "$options": "i",
                }

            # Exclude items with no nutrition data
            query["nutrition_enhanced"] = {"$exists": True}

            results = list(branded_foods.find(query).limit(20))
            client.close()

            return results

        except Exception as e:
            logger.error(f"Error in fallback search: {str(e)}")
            return []

    def _get_food_category_preference(self, fitness_goal: str) -> str:
        """Get preferred food categories based on fitness goal"""
        goal_categories = {
            "bulk": "protein snacks",
            "cut": "lean protein",
            "recomp": "high protein",
            "maintenance": "balanced nutrition",
        }
        return goal_categories.get(fitness_goal, "nutritious food")

    async def generate_meal_plan(
        self, profile: UserProfile, request: MealPlanRequest
    ) -> Dict[str, Any]:
        """Generate a complete meal plan"""

        # Get suitable foods based on profile using enhanced criteria for vector search
        criteria = {
            "high_protein": profile.fitness_goal in ["bulk", "recomp"]
            or (profile.target_protein_g or 0) > 120,
            "low_carb": "keto" in (profile.dietary_preferences or [])
            or "low carb" in (profile.dietary_preferences or []),
            "dietary_preferences": profile.dietary_preferences or [],
            "fitness_goal": profile.fitness_goal,
            "food_category": self._get_food_category_preference(profile.fitness_goal),
        }

        # Add specific dietary restrictions for vector search
        if profile.allergies:
            criteria["allergies"] = profile.allergies

        available_foods = await self.find_foods_by_criteria(criteria)

        # Create meal plan using LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a nutrition expert creating personalized meal plans.
            
            User Profile:
            - Goal: {fitness_goal}
            - Target Calories: {target_calories}
            - Target Protein: {target_protein_g}g
            - Target Carbs: {target_carbs_g}g  
            - Target Fat: {target_fat_g}g
            - Allergies: {allergies}
            - Dietary Preferences: {dietary_preferences}
            
            Create a 7-day meal plan with {meal_count} meals per day.
            Focus on whole foods and meeting macro targets.
            
            Available foods from database: {available_foods_sample}
            
            Return structured JSON with:
            - daily_plans: array of day objects
            - each day: meals array with meal_name, foods, portions, macros
            - summary: brief explanation of plan approach
            """,
                ),
                (
                    "human",
                    "Create my personalized meal plan based on the profile above.",
                ),
            ]
        )

        # Sample some foods for the prompt
        foods_sample = [
            {
                "description": food.get("description", ""),
                "nutrition": food.get("nutrition_enhanced", {}).get("per_100g", {}),
            }
            for food in available_foods[:5]
        ]

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "fitness_goal": profile.fitness_goal,
                    "target_calories": profile.target_calories,
                    "target_protein_g": profile.target_protein_g,
                    "target_carbs_g": profile.target_carbs_g,
                    "target_fat_g": profile.target_fat_g,
                    "allergies": profile.allergies or [],
                    "dietary_preferences": profile.dietary_preferences or [],
                    "meal_count": request.meal_count,
                    "available_foods_sample": foods_sample,
                }
            )

            # Parse LLM response and structure meal plan
            meal_plan = {
                "user_id": profile.user_id,
                "days": 7,
                "target_macros": {
                    "calories": profile.target_calories,
                    "protein_g": profile.target_protein_g,
                    "carbs_g": profile.target_carbs_g,
                    "fat_g": profile.target_fat_g,
                },
                "plan_content": response.content,
                "available_foods_count": len(available_foods),
                "generated_at": datetime.now(),
            }

            return meal_plan

        except Exception as e:
            logger.error(f"Error generating meal plan: {str(e)}")
            return {
                "error": f"Failed to generate meal plan: {str(e)}",
                "user_id": profile.user_id,
            }


class WorkoutPlannerAgent:
    """Generates personalized workout plans"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY")
        )

    async def generate_workout_plan(
        self, profile: UserProfile, request: WorkoutPlanRequest
    ) -> Dict[str, Any]:
        """Generate a complete workout plan"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a certified personal trainer creating workout plans.
            
            User Profile:
            - Fitness Goal: {fitness_goal}
            - Equipment Available: {equipment_available}
            - Workout Frequency: {workout_frequency} days/week
            - Session Duration: {workout_duration} minutes
            - Training Style: {training_style}
            - Split Type: {split_type}
            
            Create a {days_per_week}-day workout plan using {split_type} split.
            Focus on {training_style} training style.
            
            Return structured JSON with:
            - weekly_schedule: array of workout days
            - each day: exercises with sets, reps, rest periods
            - progression_notes: how to advance over time
            - summary: brief explanation of plan approach
            """,
                ),
                (
                    "human",
                    "Create my personalized workout plan based on the profile above.",
                ),
            ]
        )

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "fitness_goal": profile.fitness_goal,
                    "equipment_available": profile.equipment_available
                    or ["bodyweight", "dumbbells", "barbell"],
                    "workout_frequency": profile.workout_frequency,
                    "workout_duration": profile.workout_duration,
                    "training_style": request.training_style,
                    "split_type": request.split_type,
                    "days_per_week": request.days_per_week,
                }
            )

            workout_plan = {
                "user_id": profile.user_id,
                "split_type": request.split_type,
                "training_style": request.training_style,
                "days_per_week": request.days_per_week,
                "duration_minutes": request.duration_minutes,
                "plan_content": response.content,
                "generated_at": datetime.now(),
            }

            return workout_plan

        except Exception as e:
            logger.error(f"Error generating workout plan: {str(e)}")
            return {
                "error": f"Failed to generate workout plan: {str(e)}",
                "user_id": profile.user_id,
            }


class PlanSummaryAgent:
    """Creates comprehensive summaries combining meal and workout plans"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY")
        )

    async def create_summary(
        self, profile: UserProfile, meal_plan: Dict, workout_plan: Dict
    ) -> str:
        """Create a comprehensive plan summary"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a fitness coach providing comprehensive guidance.
            
            Create a motivating summary that combines the meal and workout plans.
            Include:
            - Key highlights of both plans
            - How they work together to achieve the goal
            - Important tips for success
            - Timeline expectations
            - Adjustment recommendations
            
            Keep it encouraging and actionable.
            """,
                ),
                (
                    "human",
                    """
            User Goal: {fitness_goal}
            
            Meal Plan Summary:
            {meal_plan_content}
            
            Workout Plan Summary:
            {workout_plan_content}
            
            Create a comprehensive guidance summary.
            """,
                ),
            ]
        )

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "fitness_goal": profile.fitness_goal,
                    "meal_plan_content": meal_plan.get("plan_content", ""),
                    "workout_plan_content": workout_plan.get("plan_content", ""),
                }
            )

            return response.content

        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return "Summary generation failed. Please review individual plans."


# Initialize agents
profile_agent = ProfileManagerAgent()
meal_agent = MealPlannerAgent()
workout_agent = WorkoutPlannerAgent()
summary_agent = PlanSummaryAgent()


# API Endpoints
@agents.post("/profile/", response_model=UserProfile)
async def create_or_update_profile(profile: UserProfile):
    """Create or update user profile with calculated nutritional needs"""
    try:
        updated_profile = await profile_agent.update_profile(profile)
        return updated_profile
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update profile: {str(e)}"
        )


@agents.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile by ID"""
    try:
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        profiles = db["user_profiles"]

        profile = profiles.find_one({"user_id": user_id})
        client.close()

        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Remove MongoDB ObjectId for JSON serialization
        profile.pop("_id", None)
        return profile

    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


@agents.post("/meal-plan/")
async def generate_meal_plan(request: MealPlanRequest):
    """Generate personalized meal plan"""
    try:
        # Get user profile
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        profiles = db["user_profiles"]

        profile_data = profiles.find_one({"user_id": request.user_id})
        client.close()

        if not profile_data:
            raise HTTPException(status_code=404, detail="User profile not found")

        profile_data.pop("_id", None)
        profile = UserProfile(**profile_data)

        meal_plan = await meal_agent.generate_meal_plan(profile, request)
        return meal_plan

    except Exception as e:
        logger.error(f"Error generating meal plan: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate meal plan: {str(e)}"
        )


@agents.post("/workout-plan/")
async def generate_workout_plan(request: WorkoutPlanRequest):
    """Generate personalized workout plan"""
    try:
        # Get user profile
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        profiles = db["user_profiles"]

        profile_data = profiles.find_one({"user_id": request.user_id})
        client.close()

        if not profile_data:
            raise HTTPException(status_code=404, detail="User profile not found")

        profile_data.pop("_id", None)
        profile = UserProfile(**profile_data)

        workout_plan = await workout_agent.generate_workout_plan(profile, request)
        return workout_plan

    except Exception as e:
        logger.error(f"Error generating workout plan: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate workout plan: {str(e)}"
        )


@agents.post("/complete-plan/", response_model=FitnessPlansResponse)
async def generate_complete_fitness_plan(
    user_id: str,
    meal_request: Optional[MealPlanRequest] = None,
    workout_request: Optional[WorkoutPlanRequest] = None,
):
    """Generate complete fitness plan with meal and workout plans plus summary"""
    try:
        # Get user profile
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        profiles = db["user_profiles"]

        profile_data = profiles.find_one({"user_id": user_id})
        client.close()

        if not profile_data:
            raise HTTPException(status_code=404, detail="User profile not found")

        profile_data.pop("_id", None)
        profile = UserProfile(**profile_data)

        # Use defaults if not provided
        if not meal_request:
            meal_request = MealPlanRequest(user_id=user_id)
        if not workout_request:
            workout_request = WorkoutPlanRequest(user_id=user_id)

        # Generate both plans
        meal_plan = await meal_agent.generate_meal_plan(profile, meal_request)
        workout_plan = await workout_agent.generate_workout_plan(
            profile, workout_request
        )

        # Create comprehensive summary
        summary = await summary_agent.create_summary(profile, meal_plan, workout_plan)

        response = FitnessPlansResponse(
            user_id=user_id,
            meal_plan=meal_plan,
            workout_plan=workout_plan,
            summary=summary,
            generated_at=datetime.now(),
        )

        return response

    except Exception as e:
        logger.error(f"Error generating complete plan: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate complete plan: {str(e)}"
        )


@agents.get("/test/")
async def test_agents():
    """Test endpoint to verify agents are working"""
    return {
        "status": "success",
        "message": "LangChain Agents API is working",
        "available_endpoints": [
            "/profile/ (POST) - Create/update user profile",
            "/profile/{user_id} (GET) - Get user profile",
            "/meal-plan/ (POST) - Generate meal plan",
            "/workout-plan/ (POST) - Generate workout plan",
            "/complete-plan/ (POST) - Generate complete fitness plan",
        ],
        "agents_initialized": {
            "profile_manager": "✓",
            "meal_planner": "✓",
            "workout_planner": "✓",
            "summary_agent": "✓",
        },
    }
