import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="AI Fitness Planner",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
from dotenv import load_dotenv
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class FitnessAPI:
    """Client for interacting with FastAPI backend"""

    @staticmethod
    def create_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update user profile"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/v1/agents/profile/", json=profile_data, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error creating profile: {str(e)}")
            return {}

    @staticmethod
    def get_profile(user_id: str) -> Dict[str, Any]:
        """Get user profile"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/v1/agents/profile/{user_id}", timeout=30
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting profile: {str(e)}")
            return {}

    @staticmethod
    def generate_complete_plan(
        user_id: str, meal_days: int = 7, workout_days: int = 3
    ) -> Dict[str, Any]:
        """Generate complete fitness plan"""
        try:
            params = {
                "user_id": user_id,
                "meal_request": {
                    "user_id": user_id,
                    "days": meal_days,
                    "meal_count": 3,
                },
                "workout_request": {"user_id": user_id, "days_per_week": workout_days},
            }

            response = requests.post(
                f"{API_BASE_URL}/v1/agents/complete-plan/", json=params, timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error generating plan: {str(e)}")
            return {}

    @staticmethod
    def search_nutrition(query: str, limit: int = 5) -> Dict[str, Any]:
        """Search nutrition database"""
        try:
            params = {"query": query, "limit": limit}
            response = requests.get(
                f"{API_BASE_URL}/v1/nutrition/search_nutrition/",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error searching nutrition: {str(e)}")
            return {}


def main():
    st.title("🏋️‍♂️ AI Fitness Planner")
    st.markdown("### Personalized Meal & Workout Plans Powered by LangChain Agents")

    # Initialize session state
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_123"  # Default user ID
    if "profile_created" not in st.session_state:
        st.session_state.profile_created = False
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = {}

    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Choose a page:",
            [
                "🏠 Home",
                "👤 Profile Setup",
                "🍽️ Meal Plans",
                "💪 Workout Plans",
                "📊 Complete Plan",
                "🔍 Food Search",
            ],
        )

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "👤 Profile Setup":
        show_profile_page()
    elif page == "🍽️ Meal Plans":
        show_meal_plans_page()
    elif page == "💪 Workout Plans":
        show_workout_plans_page()
    elif page == "📊 Complete Plan":
        show_complete_plan_page()
    elif page == "🔍 Food Search":
        show_food_search_page()


def show_home_page():
    """Home page with overview"""
    st.markdown(
        """
    ## Welcome to Your AI Fitness Journey! 🚀
    
    This application uses **LangChain Agents** to create personalized fitness and nutrition plans tailored specifically for you.
    
    ### 🤖 How It Works:
    
    1. **Profile Manager Agent** - Analyzes your goals, body metrics, and preferences
    2. **Meal Planner Agent** - Creates nutrition plans using our USDA food database
    3. **Workout Planner Agent** - Designs training programs based on your goals
    4. **Summary Agent** - Combines everything into actionable guidance
    
    ### 🎯 Features:
    - **Personalized Macro Calculations** based on your goals (cut/bulk/maintenance)
    - **Smart Food Recommendations** from 300k+ USDA branded foods
    - **Adaptive Workout Plans** for strength, hypertrophy, or endurance
    - **Real-time Plan Generation** using GPT-4 powered agents
    
    ### 🚀 Get Started:
    1. Set up your profile in the sidebar
    2. Generate your complete fitness plan
    3. Follow your personalized recommendations
    
    ---
    **Powered by:** FastAPI + MongoDB + LangChain + Streamlit
    """
    )

    # Quick stats from the database
    st.subheader("📈 Platform Stats")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Available Foods", "300k+", "USDA Database")

    with col2:
        st.metric("Workout Styles", "3+", "Strength, Hypertrophy, Endurance")

    with col3:
        st.metric("Plan Types", "5+", "Cut, Bulk, Maintenance, Recomp")

    with col4:
        st.metric("AI Agents", "4", "LangChain Powered")


def show_profile_page():
    """Profile setup page"""
    st.header("👤 User Profile Setup")
    st.markdown("Let's create your personalized fitness profile!")

    # Check if profile exists
    existing_profile = FitnessAPI.get_profile(st.session_state.user_id)

    if existing_profile:
        st.success("✅ Profile found! You can update it below.")
        st.session_state.current_profile = existing_profile
    else:
        st.info("🆕 Create your profile to get started.")

    with st.form("profile_form"):
        st.subheader("Basic Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", min_value=16, max_value=80, value=existing_profile.get("age", 25)
            )

            weight = st.number_input(
                "Weight (kg)",
                min_value=40.0,
                max_value=200.0,
                value=existing_profile.get("weight", 70.0),
                step=0.5,
            )

            activity_level = st.selectbox(
                "Activity Level",
                ["sedentary", "light", "moderate", "active", "very_active"],
                index=["sedentary", "light", "moderate", "active", "very_active"].index(
                    existing_profile.get("activity_level", "moderate")
                ),
            )

        with col2:
            height = st.number_input(
                "Height (cm)",
                min_value=140.0,
                max_value=220.0,
                value=existing_profile.get("height", 175.0),
                step=0.5,
            )

            fitness_goal = st.selectbox(
                "Primary Goal",
                ["cut", "bulk", "maintenance", "recomp"],
                index=["cut", "bulk", "maintenance", "recomp"].index(
                    existing_profile.get("fitness_goal", "maintenance")
                ),
            )

            workout_frequency = st.number_input(
                "Workout Days per Week",
                min_value=1,
                max_value=7,
                value=existing_profile.get("workout_frequency", 3),
            )

        st.subheader("Preferences & Restrictions")

        allergies = st.multiselect(
            "Allergies/Intolerances",
            ["dairy", "gluten", "nuts", "shellfish", "eggs", "soy"],
            default=existing_profile.get("allergies", []),
        )

        dietary_preferences = st.multiselect(
            "Dietary Preferences",
            ["vegetarian", "vegan", "keto", "paleo", "mediterranean", "low_carb"],
            default=existing_profile.get("dietary_preferences", []),
        )

        equipment_available = st.multiselect(
            "Available Equipment",
            [
                "bodyweight",
                "dumbbells",
                "barbell",
                "resistance_bands",
                "pull_up_bar",
                "gym_access",
            ],
            default=existing_profile.get(
                "equipment_available", ["bodyweight", "dumbbells"]
            ),
        )

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)

        if submitted:
            profile_data = {
                "user_id": st.session_state.user_id,
                "age": age,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "fitness_goal": fitness_goal,
                "workout_frequency": workout_frequency,
                "allergies": allergies,
                "dietary_preferences": dietary_preferences,
                "equipment_available": equipment_available,
                "created_at": (
                    datetime.now().isoformat()
                    if not existing_profile
                    else existing_profile.get("created_at")
                ),
            }

            with st.spinner("Saving profile and calculating nutritional needs..."):
                result = FitnessAPI.create_profile(profile_data)

                if result:
                    st.success("✅ Profile saved successfully!")
                    st.session_state.profile_created = True
                    st.session_state.current_profile = result

                    # Display calculated macros
                    st.subheader("🎯 Your Calculated Targets")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Daily Calories", f"{result.get('target_calories', 0):,}"
                        )

                    with col2:
                        st.metric("Protein", f"{result.get('target_protein_g', 0)}g")

                    with col3:
                        st.metric("Carbs", f"{result.get('target_carbs_g', 0)}g")

                    with col4:
                        st.metric("Fat", f"{result.get('target_fat_g', 0)}g")

                    st.balloons()


def show_meal_plans_page():
    """Meal plans page"""
    st.header("🍽️ Meal Plans")

    if not st.session_state.get("profile_created") and not st.session_state.get(
        "current_profile"
    ):
        st.warning("⚠️ Please create your profile first!")
        return

    st.markdown("Generate personalized meal plans based on your profile and goals.")

    col1, col2 = st.columns(2)

    with col1:
        days = st.selectbox("Plan Duration", [1, 3, 7, 14], index=2)

    with col2:
        meals_per_day = st.selectbox("Meals per Day", [3, 4, 5, 6], index=0)

    if st.button("🍽️ Generate Meal Plan", use_container_width=True):
        with st.spinner("Creating your personalized meal plan..."):
            # This would call the meal plan endpoint specifically
            st.info(
                "🚧 Meal plan generation coming soon! Use the Complete Plan page for now."
            )


def show_workout_plans_page():
    """Workout plans page"""
    st.header("💪 Workout Plans")

    if not st.session_state.get("profile_created") and not st.session_state.get(
        "current_profile"
    ):
        st.warning("⚠️ Please create your profile first!")
        return

    st.markdown(
        "Generate personalized workout plans based on your goals and available equipment."
    )

    col1, col2 = st.columns(2)

    with col1:
        split_type = st.selectbox(
            "Training Split",
            ["full_body", "upper_lower", "push_pull_legs", "body_part_split"],
        )

        training_style = st.selectbox(
            "Training Style", ["strength", "hypertrophy", "endurance", "powerlifting"]
        )

    with col2:
        workout_days = st.selectbox("Days per Week", [2, 3, 4, 5, 6], index=1)

        session_duration = st.selectbox(
            "Session Duration (min)", [30, 45, 60, 75, 90], index=2
        )

    if st.button("💪 Generate Workout Plan", use_container_width=True):
        with st.spinner("Creating your personalized workout plan..."):
            # This would call the workout plan endpoint specifically
            st.info(
                "🚧 Workout plan generation coming soon! Use the Complete Plan page for now."
            )


def show_complete_plan_page():
    """Complete plan generation page"""
    st.header("📊 Complete Fitness Plan")

    if not st.session_state.get("current_profile") and not FitnessAPI.get_profile(
        st.session_state.user_id
    ):
        st.warning("⚠️ Please create your profile first!")
        if st.button("👤 Go to Profile Setup"):
            st.rerun()
        return

    st.markdown(
        "Generate your complete personalized fitness plan including both meal and workout plans!"
    )

    # Plan configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🍽️ Meal Plan Settings")
        meal_plan_days = st.selectbox("Meal Plan Duration", [3, 7, 14], index=1)

    with col2:
        st.subheader("💪 Workout Settings")
        workout_days_per_week = st.selectbox(
            "Workout Days per Week", [2, 3, 4, 5], index=1
        )

    # Generate button
    if st.button("🚀 Generate Complete Plan", use_container_width=True, type="primary"):
        with st.spinner("🤖 AI Agents are creating your personalized plan..."):
            result = FitnessAPI.generate_complete_plan(
                st.session_state.user_id, meal_plan_days, workout_days_per_week
            )

            if result:
                st.success("✅ Your complete fitness plan is ready!")

                # Display the summary
                if result.get("summary"):
                    st.subheader("🎯 Your Personalized Plan Summary")
                    st.markdown(result["summary"])

                # Meal Plan Section
                if result.get("meal_plan"):
                    st.subheader("🍽️ Meal Plan")
                    meal_plan = result["meal_plan"]

                    if meal_plan.get("target_macros"):
                        st.markdown("**Daily Targets:**")

                        col1, col2, col3, col4 = st.columns(4)

                        macros = meal_plan["target_macros"]
                        with col1:
                            st.metric("Calories", f"{macros.get('calories', 0):,}")
                        with col2:
                            st.metric("Protein", f"{macros.get('protein_g', 0)}g")
                        with col3:
                            st.metric("Carbs", f"{macros.get('carbs_g', 0)}g")
                        with col4:
                            st.metric("Fat", f"{macros.get('fat_g', 0)}g")

                    if meal_plan.get("plan_content"):
                        with st.expander("📋 Detailed Meal Plan", expanded=True):
                            st.markdown(meal_plan["plan_content"])

                # Workout Plan Section
                if result.get("workout_plan"):
                    st.subheader("💪 Workout Plan")
                    workout_plan = result["workout_plan"]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Training Style", workout_plan.get("training_style", "N/A")
                        )
                    with col2:
                        st.metric("Split Type", workout_plan.get("split_type", "N/A"))
                    with col3:
                        st.metric("Days/Week", workout_plan.get("days_per_week", "N/A"))

                    if workout_plan.get("plan_content"):
                        with st.expander("🏋️‍♂️ Detailed Workout Plan", expanded=True):
                            st.markdown(workout_plan["plan_content"])

                # Plan metadata
                if result.get("generated_at"):
                    st.caption(f"Plan generated on: {result['generated_at']}")

                st.balloons()


def show_food_search_page():
    """Food search page"""
    st.header("🔍 Food Database Search")
    st.markdown("Search through our comprehensive USDA nutrition database.")

    # Search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search for foods:",
            placeholder="e.g., chicken breast, greek yogurt, quinoa...",
        )

    with col2:
        search_limit = st.selectbox("Results", [5, 10, 20], index=1)

    if st.button("🔍 Search Foods", use_container_width=True) and search_query:
        with st.spinner("Searching nutrition database..."):
            results = FitnessAPI.search_nutrition(search_query, search_limit)

            if results and results.get("results"):
                st.success(
                    f"Found {results['results_found']} results for '{search_query}'"
                )

                for i, food in enumerate(results["results"]):
                    with st.expander(
                        f"🥘 {food.get('description', 'Unknown Food')} - {food.get('brand_owner', 'Unknown Brand')}"
                    ):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Product Info:**")
                            st.write(f"**Brand:** {food.get('brand_name', 'N/A')}")
                            st.write(
                                f"**Category:** {food.get('food_category', 'N/A')}"
                            )
                            st.write(
                                f"**Serving Size:** {food.get('serving_size', 'N/A')} {food.get('serving_size_unit', '')}"
                            )

                            if food.get("ingredients"):
                                st.write(
                                    f"**Ingredients:** {food['ingredients'][:200]}{'...' if len(food['ingredients']) > 200 else ''}"
                                )

                        with col2:
                            nutrition = food.get("nutrition_enhanced", {})
                            per_100g = nutrition.get("per_100g", {})

                            if per_100g:
                                st.markdown("**Nutrition per 100g:**")

                                # Create metrics
                                metrics_col1, metrics_col2 = st.columns(2)

                                with metrics_col1:
                                    st.metric(
                                        "Calories",
                                        f"{per_100g.get('energy_kcal', 0)} kcal",
                                    )
                                    st.metric(
                                        "Protein", f"{per_100g.get('protein_g', 0)} g"
                                    )

                                with metrics_col2:
                                    st.metric(
                                        "Carbs", f"{per_100g.get('carbs_g', 0)} g"
                                    )
                                    st.metric(
                                        "Fat", f"{per_100g.get('total_fat_g', 0)} g"
                                    )

                                # Macro breakdown if available
                                macro_breakdown = nutrition.get("macro_breakdown", {})
                                if macro_breakdown.get("primary_macro_category"):
                                    st.write(
                                        f"**Primary Macro:** {macro_breakdown['primary_macro_category'].replace('_', ' ').title()}"
                                    )
                            else:
                                st.warning("Nutrition data not available")
            else:
                st.warning("No results found. Try a different search term!")


if __name__ == "__main__":
    main()
