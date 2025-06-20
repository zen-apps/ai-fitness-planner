import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fitness_agents")

agents = APIRouter()

AVAILABLE_FOODS_COUNT = 100  # Number of foods to return in search results


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
    meal_count: Optional[int] = 5
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


# Structured Output Models for LLM Responses
class MealMacros(BaseModel):
    calories: float = Field(description="Total calories for the meal")
    protein_g: float = Field(description="Protein in grams")
    carbs_g: float = Field(description="Carbohydrates in grams")
    fat_g: float = Field(description="Fat in grams")


class MealFood(BaseModel):
    food_name: str = Field(description="Name of the food item")
    portion: str = Field(description="Serving size (e.g., '150g', '1 cup', '1 medium')")
    calories: float = Field(description="Calories for this portion")
    protein_g: float = Field(description="Protein in grams for this portion")
    carbs_g: float = Field(description="Carbs in grams for this portion")
    fat_g: float = Field(description="Fat in grams for this portion")


class Meal(BaseModel):
    meal_name: str = Field(
        description="Name of the meal (e.g., 'Breakfast', 'Snack 1', 'Lunch', 'Snack 2', 'Dinner')"
    )
    foods: List[MealFood] = Field(description="List of foods in this meal")
    total_macros: MealMacros = Field(description="Total macros for the entire meal")
    preparation_notes: Optional[str] = Field(
        description="Brief preparation instructions", default=None
    )


class DailyMealPlan(BaseModel):
    day: int = Field(description="Day number (1-7)")
    day_name: str = Field(description="Day of the week")
    meals: List[Meal] = Field(description="List of meals for this day")
    daily_totals: MealMacros = Field(description="Total macros for the entire day")


class MealPlanStructured(BaseModel):
    plan_name: str = Field(description="Name/title for this meal plan")
    days: List[DailyMealPlan] = Field(description="7-day meal plan")
    target_macros: MealMacros = Field(description="Daily target macros")
    key_principles: List[str] = Field(
        description="Key nutritional principles followed in this plan"
    )
    shopping_tips: List[str] = Field(
        description="Important shopping and preparation tips"
    )


class Exercise(BaseModel):
    exercise_name: str = Field(description="Name of the exercise")
    sets: int = Field(description="Number of sets")
    reps: str = Field(
        description="Number of reps (can be range like '8-12' or specific number)"
    )
    rest_seconds: int = Field(description="Rest time between sets in seconds")
    notes: Optional[str] = Field(description="Form cues or modifications", default=None)


class WorkoutDay(BaseModel):
    day: int = Field(description="Day number in the weekly schedule")
    day_name: str = Field(description="Name of the workout day")
    focus: str = Field(
        description="Main focus of this workout (e.g., 'Upper Body', 'Push', 'Full Body')"
    )
    exercises: List[Exercise] = Field(description="List of exercises for this day")
    estimated_duration: int = Field(description="Estimated workout duration in minutes")
    warm_up: List[str] = Field(description="Warm-up exercises/activities")
    cool_down: List[str] = Field(description="Cool-down exercises/stretches")


class WorkoutPlanStructured(BaseModel):
    plan_name: str = Field(description="Name/title for this workout plan")
    split_type: str = Field(description="Type of training split used")
    training_style: str = Field(description="Primary training style/goal")
    weekly_schedule: List[WorkoutDay] = Field(description="Weekly workout schedule")
    progression_strategy: str = Field(description="How to progress over time")
    equipment_needed: List[str] = Field(description="Equipment required for this plan")
    key_principles: List[str] = Field(
        description="Important training principles to follow"
    )


class ComprehensiveSummary(BaseModel):
    overview: str = Field(
        description="High-level overview of the complete fitness plan"
    )
    key_highlights: List[str] = Field(
        description="Main highlights combining both meal and workout plans"
    )
    synergy_explanation: str = Field(
        description="How the meal and workout plans work together"
    )
    success_tips: List[str] = Field(description="Important tips for achieving success")
    timeline_expectations: str = Field(description="What to expect and when")
    adjustment_recommendations: List[str] = Field(
        description="When and how to adjust the plans"
    )
    motivation_message: str = Field(description="Encouraging closing message")


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

    def __init__(self, use_o3_mini: bool = True):
        if use_o3_mini:
            self.llm = ChatOpenAI(
                model="o3-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.7, 
                api_key=os.getenv("OPENAI_API_KEY")
            )

    @traceable(name="find_foods_by_criteria")
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
                limit=AVAILABLE_FOODS_COUNT,
                similarity_threshold=0.5,
            )

            # Use existing vector search
            search_result = await search_nutrition_semantic(nutrition_query)

            # Convert results to expected format with source attribution
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
                    "search_source": "search_nutrition_semantic",  # Source attribution
                }
                foods.append(food_item)

            # Create summary for LangSmith tracing
            trace_data = {
                "criteria": criteria,
                "search_query": search_query,
                "search_source": "search_nutrition_semantic",
                "foods_found": len(foods),
                "search_time_ms": search_result.search_time_ms,
                "results_summary": [
                    {
                        "description": food.get("description", "")[:50],
                        "similarity_score": food.get("similarity_score", 0),
                        "protein_per_100g": food.get("nutrition_enhanced", {})
                        .get("per_100g", {})
                        .get("protein_g", 0),
                    }
                    for food in foods[
                        :AVAILABLE_FOODS_COUNT
                    ]  # First 5 results for tracing
                ],
            }

            # Log detailed trace data
            logger.info(f"Vector search successful: {trace_data}")

            return foods

        except Exception as e:
            logger.error(f"Error in vector food search: {str(e)}")
            # Fallback to basic MongoDB query if vector search fails
            return await self._fallback_mongo_search(criteria)

    @traceable(name="fallback_mongo_search")
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

            # Add source attribution to each result
            for result in results:
                result["search_source"] = "fallback_mongo_search"

            # Create trace data for LangSmith
            trace_data = {
                "criteria": criteria,
                "search_source": "fallback_mongo_search",
                "mongo_query": query,
                "foods_found": len(results),
                "results_summary": [
                    {
                        "description": result.get("description", "")[:50],
                        "brand": result.get("brandOwner", "")[:30],
                        "protein_per_100g": result.get("nutrition_enhanced", {})
                        .get("per_100g", {})
                        .get("protein_g", 0),
                    }
                    for result in results[:5]  # First 5 results for tracing
                ],
            }

            logger.info(f"Fallback MongoDB search: {trace_data}")

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

    @traceable(name="generate_meal_plan")
    async def generate_meal_plan(
        self, profile: UserProfile, request: MealPlanRequest
    ) -> Dict[str, Any]:
        """Generate a complete meal plan using structured output"""

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

        # Sample some foods for the prompt
        foods_sample = [
            {
                "description": food.get("description", ""),
                "nutrition": food.get("nutrition_enhanced", {}).get("per_100g", {}),
                "brand": food.get("brandOwner", ""),
            }
            for food in available_foods[
                :AVAILABLE_FOODS_COUNT
            ]  # Use more foods for better variety
        ]

        # Create meal plan using structured LLM
        messages = [
            SystemMessage(
                content=f"""You are a nutrition expert creating personalized meal plans.
                
                User Profile:
                - Goal: {profile.fitness_goal}
                - Target Calories: {profile.target_calories}
                - Target Protein: {profile.target_protein_g}g
                - Target Carbs: {profile.target_carbs_g}g  
                - Target Fat: {profile.target_fat_g}g
                - Allergies: {profile.allergies or []}
                - Dietary Preferences: {profile.dietary_preferences or []}
                
                Create a complete 7-day meal plan with {request.meal_count} meals per day.
                Two of the meals (Snack 1 and Snack 2) should be high protein snacks.
                Focus on whole foods, meeting macro targets, and creating variety.
                Use the available foods from the database when possible, but supplement with common whole foods.
                Maximum of 5 different foods per meal.
                
                Calculate accurate portions and macros for each food item.
                Ensure daily totals align closely with target macros.
                Provide practical preparation notes and shopping tips."""
            ),
            HumanMessage(
                content=f"""Create my personalized meal plan based on the profile above.
                
                Available foods from database: {foods_sample}
                
                Create a structured meal plan that meets my nutritional goals."""
            ),
        ]

        try:
            structured_llm = self.llm.with_structured_output(MealPlanStructured)
            meal_plan_structured = structured_llm.invoke(messages)

            # Convert structured output to dictionary format
            meal_plan = {
                "user_id": profile.user_id,
                "plan_name": meal_plan_structured.plan_name,
                "days": len(meal_plan_structured.days),
                "target_macros": {
                    "calories": meal_plan_structured.target_macros.calories,
                    "protein_g": meal_plan_structured.target_macros.protein_g,
                    "carbs_g": meal_plan_structured.target_macros.carbs_g,
                    "fat_g": meal_plan_structured.target_macros.fat_g,
                },
                "daily_plans": [day.dict() for day in meal_plan_structured.days],
                "key_principles": meal_plan_structured.key_principles,
                "shopping_tips": meal_plan_structured.shopping_tips,
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

    @traceable(name="generate_workout_plan")
    async def generate_workout_plan(
        self, profile: UserProfile, request: WorkoutPlanRequest
    ) -> Dict[str, Any]:
        """Generate a complete workout plan using structured output"""

        messages = [
            SystemMessage(
                content=f"""You are a certified personal trainer creating comprehensive workout plans.
                
                User Profile:
                - Fitness Goal: {profile.fitness_goal}
                - Equipment Available: {profile.equipment_available or ['bodyweight', 'dumbbells', 'barbell']}
                - Workout Frequency: {profile.workout_frequency} days/week
                - Session Duration: {profile.workout_duration} minutes
                - Training Style: {request.training_style}
                - Split Type: {request.split_type}
                
                Create a {request.days_per_week}-day workout plan using {request.split_type} split.
                Focus on {request.training_style} training with proper progression.
                
                Include:
                - Specific exercises with sets, reps, and rest periods
                - Warm-up and cool-down for each session
                - Clear progression strategy
                - Equipment requirements
                - Key training principles
                
                Ensure workouts fit within the specified duration and use available equipment."""
            ),
            HumanMessage(
                content="Create my personalized workout plan based on the profile above."
            ),
        ]

        try:
            structured_llm = self.llm.with_structured_output(WorkoutPlanStructured)
            workout_plan_structured = structured_llm.invoke(messages)

            # Convert structured output to dictionary format
            workout_plan = {
                "user_id": profile.user_id,
                "plan_name": workout_plan_structured.plan_name,
                "split_type": workout_plan_structured.split_type,
                "training_style": workout_plan_structured.training_style,
                "days_per_week": request.days_per_week,
                "duration_minutes": request.duration_minutes,
                "weekly_schedule": [
                    day.dict() for day in workout_plan_structured.weekly_schedule
                ],
                "progression_strategy": workout_plan_structured.progression_strategy,
                "equipment_needed": workout_plan_structured.equipment_needed,
                "key_principles": workout_plan_structured.key_principles,
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

    @traceable(name="create_summary")
    async def create_summary(
        self, profile: UserProfile, meal_plan: Dict, workout_plan: Dict
    ) -> str:
        """Create a comprehensive plan summary using structured output"""

        messages = [
            SystemMessage(
                content="""You are a fitness coach providing comprehensive guidance that combines nutrition and training.
                
                Create a motivating and actionable summary that explains how the meal and workout plans work together.
                Include practical advice, realistic timelines, and encouragement.
                Focus on the synergy between nutrition and training for optimal results."""
            ),
            HumanMessage(
                content=f"""User Goal: {profile.fitness_goal}
                
                Meal Plan Overview:
                - Plan Name: {meal_plan.get('plan_name', 'Custom Meal Plan')}
                - Key Principles: {meal_plan.get('key_principles', [])}
                - Target Macros: {meal_plan.get('target_macros', {})}
                
                Workout Plan Overview:
                - Plan Name: {workout_plan.get('plan_name', 'Custom Workout Plan')}
                - Training Style: {workout_plan.get('training_style', '')}
                - Split Type: {workout_plan.get('split_type', '')}
                - Key Principles: {workout_plan.get('key_principles', [])}
                
                Create a comprehensive guidance summary that shows how these plans work together."""
            ),
        ]

        try:
            structured_llm = self.llm.with_structured_output(ComprehensiveSummary)
            summary_structured = structured_llm.invoke(messages)

            # Convert to formatted string
            summary_text = f"""
### Overview
{summary_structured.overview}

### Key Highlights
{chr(10).join(f"• {highlight}" for highlight in summary_structured.key_highlights)}

### How Your Plans Work Together
{summary_structured.synergy_explanation}

### Success Tips
{chr(10).join(f"• {tip}" for tip in summary_structured.success_tips)}

### What to Expect
{summary_structured.timeline_expectations}

### When to Adjust
{chr(10).join(f"• {rec}" for rec in summary_structured.adjustment_recommendations)}

### Your Journey Ahead
{summary_structured.motivation_message}
            """.strip()

            return summary_text

        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return "Summary generation failed. Please review individual plans for detailed guidance."


# Initialize agents
profile_agent = ProfileManagerAgent()
meal_agent = MealPlannerAgent(use_o3_mini=True)  # Default to O3-mini
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
