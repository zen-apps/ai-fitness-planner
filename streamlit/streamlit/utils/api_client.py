import streamlit as st
import requests
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/dev.env")

# Set API base URL with proper fallbacks
API_BASE_URL = (
    os.getenv("API_BASE_URL") or os.getenv("BACKEND_HOST") or "http://localhost:8000"
)

# Clean up the URL - remove quotes and trailing slash
API_BASE_URL = API_BASE_URL.strip("\"'").rstrip("/")

# Ensure we have a proper protocol
if not API_BASE_URL.startswith(("http://", "https://")):
    API_BASE_URL = f"http://{API_BASE_URL}"


class FitnessAPI:
    """Client for interacting with FastAPI backend"""

    @staticmethod
    def get_api_url() -> str:
        """Get the current API URL (with override support)"""
        return st.session_state.get("api_url_override", API_BASE_URL)

    @staticmethod
    def test_connection() -> Dict[str, Any]:
        """Test if the API is reachable"""
        api_url = FitnessAPI.get_api_url()
        try:
            response = requests.get(f"{api_url}/docs#/", timeout=10)
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "url": f"{api_url}/docs#/",
                "error": None,
            }
        except requests.exceptions.ConnectTimeout:
            return {
                "success": False,
                "error": "Connection timeout",
                "url": f"{api_url}/docs#/",
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection refused",
                "url": f"{api_url}/docs#/",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "url": f"{api_url}/docs#/"}

    @staticmethod
    def create_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update user profile"""
        api_url = FitnessAPI.get_api_url()
        try:
            response = requests.post(
                f"{api_url}/v1/agents/profile/", json=profile_data, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error creating profile: {str(e)}")
            return {}

    @staticmethod
    def get_profile(user_id: str) -> Dict[str, Any]:
        """Get user profile"""
        api_url = FitnessAPI.get_api_url()
        try:
            response = requests.get(
                f"{api_url}/v1/agents/profile/{user_id}", timeout=30
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting profile: {str(e)}")
            return {}

    @staticmethod
    def generate_complete_plan(user_id: str, workout_days: int = 3) -> Dict[str, Any]:
        """Generate complete fitness plan"""
        api_url = FitnessAPI.get_api_url()
        try:
            params = {
                "user_id": user_id,
                "meal_request": {
                    "user_id": user_id,
                    "meal_count": 3,
                },
                "workout_request": {"user_id": user_id, "days_per_week": workout_days},
            }

            response = requests.post(
                f"{api_url}/v1/agents/complete-plan/", json=params, timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error generating plan: {str(e)}")
            return {}

    @staticmethod
    def generate_langgraph_plan(user_id: str, use_o3_mini: bool = True, use_full_database: bool = False) -> Dict[str, Any]:
        """Generate fitness plan using LangGraph workflow"""
        api_url = FitnessAPI.get_api_url()
        try:
            request_data = {
                "user_id": user_id,
                "generate_meal_plan": True,
                "generate_workout_plan": True,
                "meal_preferences": {"meal_count": 3},
                "workout_preferences": {},  # Will use profile settings
                "use_o3_mini": use_o3_mini,
                "use_full_database": use_full_database,
            }

            response = requests.post(
                f"{api_url}/v1/langgraph/generate-fitness-plan/",
                json=request_data,
                timeout=360,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error generating LangGraph plan: {str(e)}")
            return {}

    @staticmethod
    def search_nutrition(query: str, limit: int = 5) -> Dict[str, Any]:
        """Search nutrition database using vector search"""
        api_url = FitnessAPI.get_api_url()
        try:
            request_data = {"query": query, "limit": limit, "similarity_threshold": 0.6}
            response = requests.post(
                f"{api_url}/v1/nutrition_search/search_nutrition_semantic/",
                json=request_data,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error searching nutrition: {str(e)}")
            return {}

    @staticmethod
    def check_database_availability() -> Dict[str, Any]:
        """Check which databases are available (full vs sample)"""
        api_url = FitnessAPI.get_api_url()
        try:
            response = requests.get(
                f"{api_url}/v1/nutrition_setup/database_availability/",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error checking database availability: {str(e)}")
            return {}


def init_session_state():
    """Initialize session state variables"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_123"  # Default user ID
    if "profile_created" not in st.session_state:
        st.session_state.profile_created = False
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = {}
    if "api_url_override" not in st.session_state:
        st.session_state.api_url_override = API_BASE_URL


def setup_api_settings_sidebar():
    """Setup API settings in sidebar"""
    with st.sidebar:
        with st.expander("🔧 API Settings"):
            new_url = st.text_input(
                "Override API URL:", value=st.session_state.api_url_override
            )
            if st.button("Update URL"):
                st.session_state.api_url_override = new_url.strip().rstrip("/")
                st.success("URL updated!")
                st.rerun()

        # Show current API URL
        current_url = FitnessAPI.get_api_url()
