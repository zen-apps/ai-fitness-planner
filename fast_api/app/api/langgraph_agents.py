import os
import json
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from pymongo import MongoClient

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import existing agent classes
from .agents import (
    ProfileManagerAgent, 
    MealPlannerAgent, 
    WorkoutPlannerAgent, 
    PlanSummaryAgent,
    UserProfile,
    MealPlanRequest,
    WorkoutPlanRequest,
    get_mongo_client
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("langgraph_fitness_agents")

langgraph_agents = APIRouter()

# LangGraph State Definition
class FitnessState(TypedDict):
    """State for the fitness planning workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    user_profile: Optional[Dict[str, Any]]
    meal_plan: Optional[Dict[str, Any]]
    workout_plan: Optional[Dict[str, Any]]
    summary: Optional[str]
    current_step: str
    errors: List[str]
    preferences: Dict[str, Any]

# Pydantic models for API
class LangGraphFitnessRequest(BaseModel):
    user_id: str
    user_profile: Optional[UserProfile] = None
    meal_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    workout_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    generate_meal_plan: bool = True
    generate_workout_plan: bool = True
    days: Optional[int] = 7
    
class LangGraphFitnessResponse(BaseModel):
    user_id: str
    workflow_status: str
    user_profile: Optional[Dict[str, Any]] = None
    meal_plan: Optional[Dict[str, Any]] = None
    workout_plan: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    execution_steps: List[str] = []
    errors: List[str] = []
    generated_at: datetime

class FitnessWorkflow:
    """LangGraph workflow orchestrator for fitness planning"""
    
    def __init__(self):
        self.profile_agent = ProfileManagerAgent()
        self.meal_agent = MealPlannerAgent()
        self.workout_agent = WorkoutPlannerAgent()
        self.summary_agent = PlanSummaryAgent()
        
        # Initialize LLM for coordination
        self.coordinator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(FitnessState)
        
        # Add nodes for each step
        workflow.add_node("profile_manager", self._manage_profile)
        workflow.add_node("meal_planner", self._plan_meals)
        workflow.add_node("workout_planner", self._plan_workout)
        workflow.add_node("plan_coordinator", self._coordinate_plans)
        workflow.add_node("summary_generator", self._generate_summary)
        
        # Define the workflow edges
        workflow.set_entry_point("profile_manager")
        
        # Conditional routing based on preferences
        workflow.add_conditional_edges(
            "profile_manager",
            self._route_after_profile,
            {
                "meal_only": "meal_planner",
                "workout_only": "workout_planner", 
                "both": "plan_coordinator",
                "error": END
            }
        )
        
        workflow.add_edge("plan_coordinator", "meal_planner")
        workflow.add_edge("meal_planner", "workout_planner")
        workflow.add_edge("workout_planner", "summary_generator")
        workflow.add_edge("summary_generator", END)
        
        return workflow.compile()
    
    async def _manage_profile(self, state: FitnessState) -> FitnessState:
        """Node: Manage user profile and calculate nutritional needs"""
        logger.info(f"Managing profile for user: {state['user_id']}")
        
        try:
            state["current_step"] = "profile_management"
            
            # Get or create user profile
            client = get_mongo_client()
            db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
            profiles = db["user_profiles"]
            
            existing_profile = profiles.find_one({"user_id": state["user_id"]})
            
            if existing_profile:
                existing_profile.pop("_id", None)
                profile = UserProfile(**existing_profile)
                logger.info("Loaded existing user profile")
            else:
                # Create new profile with defaults
                profile = UserProfile(
                    user_id=state["user_id"],
                    age=30,
                    weight=70.0,
                    height=175.0,
                    activity_level="moderate",
                    fitness_goal="maintenance"
                )
                logger.info("Created new user profile with defaults")
            
            # Update profile with calculated values
            updated_profile = await self.profile_agent.update_profile(profile)
            
            state["user_profile"] = updated_profile.dict()
            state["messages"].append(
                SystemMessage(content=f"User profile updated with target: {updated_profile.target_calories} calories")
            )
            
            client.close()
            
        except Exception as e:
            error_msg = f"Profile management error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
        return state
    
    def _route_after_profile(self, state: FitnessState) -> str:
        """Conditional routing after profile management"""
        if state["errors"]:
            return "error"
        
        preferences = state.get("preferences", {})
        generate_meal = preferences.get("generate_meal_plan", True)
        generate_workout = preferences.get("generate_workout_plan", True)
        
        if generate_meal and generate_workout:
            return "both"
        elif generate_meal:
            return "meal_only"
        elif generate_workout:
            return "workout_only"
        else:
            return "both"  # Default to both if unclear
    
    async def _plan_meals(self, state: FitnessState) -> FitnessState:
        """Node: Generate meal plan"""
        logger.info(f"Planning meals for user: {state['user_id']}")
        
        try:
            state["current_step"] = "meal_planning"
            
            if not state.get("user_profile"):
                raise ValueError("User profile not available for meal planning")
            
            profile = UserProfile(**state["user_profile"])
            
            # Create meal plan request
            meal_preferences = state.get("preferences", {}).get("meal_preferences", {})
            request = MealPlanRequest(
                user_id=state["user_id"],
                days=meal_preferences.get("days", 7),
                meal_count=meal_preferences.get("meal_count", 3)
            )
            
            meal_plan = await self.meal_agent.generate_meal_plan(profile, request)
            state["meal_plan"] = meal_plan
            
            state["messages"].append(
                SystemMessage(content=f"Meal plan generated for {request.days} days")
            )
            
        except Exception as e:
            error_msg = f"Meal planning error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
        return state
    
    async def _plan_workout(self, state: FitnessState) -> FitnessState:
        """Node: Generate workout plan"""
        logger.info(f"Planning workouts for user: {state['user_id']}")
        
        try:
            state["current_step"] = "workout_planning"
            
            if not state.get("user_profile"):
                raise ValueError("User profile not available for workout planning")
            
            profile = UserProfile(**state["user_profile"])
            
            # Create workout plan request
            workout_preferences = state.get("preferences", {}).get("workout_preferences", {})
            request = WorkoutPlanRequest(
                user_id=state["user_id"],
                split_type=workout_preferences.get("split_type", "full_body"),
                training_style=workout_preferences.get("training_style", "hypertrophy"),
                days_per_week=workout_preferences.get("days_per_week", 3)
            )
            
            workout_plan = await self.workout_agent.generate_workout_plan(profile, request)
            state["workout_plan"] = workout_plan
            
            state["messages"].append(
                SystemMessage(content=f"Workout plan generated: {request.split_type} split")
            )
            
        except Exception as e:
            error_msg = f"Workout planning error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
        return state
    
    async def _coordinate_plans(self, state: FitnessState) -> FitnessState:
        """Node: Coordinate between meal and workout planning"""
        logger.info(f"Coordinating plans for user: {state['user_id']}")
        
        try:
            state["current_step"] = "plan_coordination"
            
            # Use LLM to coordinate timing and interactions between plans
            coordination_prompt = f"""
            You are coordinating meal and workout plans for optimal results.
            
            User Goal: {state.get('user_profile', {}).get('fitness_goal', 'maintenance')}
            
            Consider:
            - Pre/post workout nutrition timing
            - Rest day meal adjustments
            - Macro distribution around training
            - Recovery nutrition needs
            
            Provide coordination insights as a brief message.
            """
            
            coordination_message = await self.coordinator_llm.ainvoke([
                SystemMessage(content=coordination_prompt)
            ])
            
            state["messages"].append(
                SystemMessage(content=f"Plans coordinated: {coordination_message.content}")
            )
            
        except Exception as e:
            error_msg = f"Plan coordination error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
        return state
    
    async def _generate_summary(self, state: FitnessState) -> FitnessState:
        """Node: Generate comprehensive summary"""
        logger.info(f"Generating summary for user: {state['user_id']}")
        
        try:
            state["current_step"] = "summary_generation"
            
            if not state.get("user_profile"):
                raise ValueError("User profile not available for summary")
            
            profile = UserProfile(**state["user_profile"])
            meal_plan = state.get("meal_plan", {})
            workout_plan = state.get("workout_plan", {})
            
            summary = await self.summary_agent.create_summary(
                profile, meal_plan, workout_plan
            )
            
            state["summary"] = summary
            state["messages"].append(
                SystemMessage(content="Comprehensive fitness plan summary generated")
            )
            
        except Exception as e:
            error_msg = f"Summary generation error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
        return state
    
    async def execute_workflow(self, request: LangGraphFitnessRequest) -> Dict[str, Any]:
        """Execute the complete fitness planning workflow"""
        
        # Initialize state
        initial_state: FitnessState = {
            "messages": [HumanMessage(content=f"Generate fitness plan for user {request.user_id}")],
            "user_id": request.user_id,
            "user_profile": None,
            "meal_plan": None,
            "workout_plan": None,
            "summary": None,
            "current_step": "initialization",
            "errors": [],
            "preferences": {
                "generate_meal_plan": request.generate_meal_plan,
                "generate_workout_plan": request.generate_workout_plan,
                "meal_preferences": request.meal_preferences,
                "workout_preferences": request.workout_preferences
            }
        }
        
        # If user profile provided, update it first
        if request.user_profile:
            request.user_profile.user_id = request.user_id
            await self.profile_agent.update_profile(request.user_profile)
        
        # Execute workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Track execution steps
            execution_steps = [msg.content for msg in final_state["messages"] if isinstance(msg, SystemMessage)]
            
            return {
                "user_id": request.user_id,
                "workflow_status": "completed" if not final_state["errors"] else "completed_with_errors",
                "user_profile": final_state.get("user_profile"),
                "meal_plan": final_state.get("meal_plan"),
                "workout_plan": final_state.get("workout_plan"),
                "summary": final_state.get("summary"),
                "execution_steps": execution_steps,
                "errors": final_state.get("errors", []),
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {
                "user_id": request.user_id,
                "workflow_status": "failed",
                "errors": [f"Workflow execution failed: {str(e)}"],
                "generated_at": datetime.now()
            }

# Initialize workflow
fitness_workflow = FitnessWorkflow()

# API Endpoints
@langgraph_agents.post("/generate-fitness-plan/", response_model=LangGraphFitnessResponse)
async def generate_fitness_plan_workflow(request: LangGraphFitnessRequest):
    """Generate complete fitness plan using LangGraph workflow orchestration"""
    try:
        result = await fitness_workflow.execute_workflow(request)
        return LangGraphFitnessResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in fitness plan workflow: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to execute fitness plan workflow: {str(e)}"
        )

@langgraph_agents.post("/quick-plan/")
async def generate_quick_plan(
    user_id: str,
    age: int = 30,
    weight: float = 70.0,
    height: float = 175.0,
    fitness_goal: str = "maintenance",
    days: int = 7
):
    """Quick fitness plan generation with minimal input"""
    try:
        # Create profile
        profile = UserProfile(
            user_id=user_id,
            age=age,
            weight=weight,
            height=height,
            fitness_goal=fitness_goal,
            activity_level="moderate"
        )
        
        # Create request
        request = LangGraphFitnessRequest(
            user_id=user_id,
            user_profile=profile,
            days=days,
            generate_meal_plan=True,
            generate_workout_plan=True
        )
        
        result = await fitness_workflow.execute_workflow(request)
        return result
        
    except Exception as e:
        logger.error(f"Error in quick plan generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate quick plan: {str(e)}"
        )

@langgraph_agents.get("/workflow-status/{user_id}")
async def get_workflow_status(user_id: str):
    """Get the latest workflow execution status for a user"""
    try:
        # This could be enhanced to store workflow executions in MongoDB
        # For now, return a simple status check
        
        client = get_mongo_client()
        db = client[os.getenv("MONGO_DB_NAME", "usda_nutrition")]
        profiles = db["user_profiles"]
        
        profile = profiles.find_one({"user_id": user_id})
        client.close()
        
        if profile:
            return {
                "user_id": user_id,
                "profile_exists": True,
                "last_updated": profile.get("updated_at"),
                "workflow_available": True
            }
        else:
            return {
                "user_id": user_id,
                "profile_exists": False,
                "workflow_available": True
            }
            
    except Exception as e:
        logger.error(f"Error checking workflow status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check workflow status: {str(e)}"
        )

@langgraph_agents.get("/test-vector-search/")
async def test_vector_search():
    """Test the vector search integration"""
    try:
        from .agents import MealPlannerAgent
        
        meal_agent = MealPlannerAgent()
        
        # Test criteria for high protein foods
        test_criteria = {
            "high_protein": True,
            "fitness_goal": "bulk",
            "food_category": "protein snacks"
        }
        
        foods = await meal_agent.find_foods_by_criteria(test_criteria)
        
        return {
            "test_status": "success",
            "criteria_tested": test_criteria,
            "foods_found": len(foods),
            "sample_foods": [
                {
                    "description": food.get("description", ""),
                    "brand": food.get("brandOwner", ""),
                    "similarity_score": food.get("similarity_score", 0),
                    "protein_per_100g": food.get("nutrition_enhanced", {}).get("per_100g", {}).get("protein_g", 0)
                }
                for food in foods[:3]  # Show first 3 results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error testing vector search: {str(e)}")
        return {
            "test_status": "failed", 
            "error": str(e)
        }

@langgraph_agents.get("/test-workflow/")
async def test_workflow():
    """Test the LangGraph workflow with a sample user"""
    try:
        test_request = LangGraphFitnessRequest(
            user_id="test_user_123",
            user_profile=UserProfile(
                user_id="test_user_123",
                age=25,
                weight=75.0,
                height=180.0,
                fitness_goal="bulk",
                activity_level="active"
            ),
            days=3,
            generate_meal_plan=True,
            generate_workout_plan=True
        )
        
        result = await fitness_workflow.execute_workflow(test_request)
        
        # Return a summary without the full plans for testing
        return {
            "test_status": "success",
            "workflow_status": result["workflow_status"],
            "steps_executed": len(result["execution_steps"]),
            "errors_count": len(result["errors"]),
            "has_meal_plan": result["meal_plan"] is not None,
            "has_workout_plan": result["workout_plan"] is not None,
            "has_summary": result["summary"] is not None,
            "generated_at": result["generated_at"]
        }
        
    except Exception as e:
        logger.error(f"Error in workflow test: {str(e)}")
        return {
            "test_status": "failed",
            "error": str(e)
        }