import streamlit as st
import requests
import os
from utils.api_client import FitnessAPI, init_session_state, setup_api_settings_sidebar, API_BASE_URL

# Configure page
st.set_page_config(
    page_title="API Testing - AI Fitness Planner",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

st.header("🧪 API Testing & Debugging")
st.markdown("Test all API endpoints and troubleshoot connection issues.")

# Connection status
st.subheader("🔗 Connection Status")
col1, col2 = st.columns(2)

with col1:
    current_url = FitnessAPI.get_api_url()
    st.code(f"API Base URL: {current_url}")

with col2:
    if st.button("🧪 Test Connection"):
        with st.spinner("Testing connection..."):
            result = FitnessAPI.test_connection()
            if result["success"]:
                st.success("✅ API connection successful!")
                st.write(f"Status Code: {result.get('status_code', 'N/A')}")
            else:
                st.error("❌ API connection failed!")
                st.write(f"Error: {result.get('error', 'Unknown')}")
                st.write(f"URL: {result.get('url', 'N/A')}")

# Quick URL tests
st.subheader("🔍 Quick URL Tests")
test_urls = [
    "http://34.70.25.107:1015",
    "http://localhost:1015",
    "http://localhost:8000",
    "http://127.0.0.1:1015",
]

if st.button("🧪 Test Common URLs"):
    for url in test_urls:
        try:
            response = requests.get(f"{url}/docs#/", timeout=5)
            if response.status_code == 200:
                st.success(f"✅ {url} - Working!")
            else:
                st.warning(f"⚠️ {url} - Status: {response.status_code}")
        except Exception as e:
            st.error(f"❌ {url} - Error: {str(e)[:50]}...")

# Test individual endpoints
st.subheader("🔧 Endpoint Testing")

# Test vector search
if st.button("Test Vector Search"):
    with st.spinner("Testing vector search..."):
        try:
            current_url = FitnessAPI.get_api_url()
            response = requests.get(
                f"{current_url}/v1/langgraph/test-vector-search/", timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Vector search working!")
                st.json(result)
            else:
                st.error(f"❌ Vector search failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Vector search error: {str(e)}")

# Test LangGraph workflow
if st.button("Test LangGraph Workflow"):
    with st.spinner("Testing LangGraph workflow..."):
        try:
            current_url = FitnessAPI.get_api_url()
            response = requests.get(
                f"{current_url}/v1/langgraph/test-workflow/", timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                st.success("✅ LangGraph workflow working!")
                st.json(result)
            else:
                st.error(f"❌ LangGraph workflow failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ LangGraph workflow error: {str(e)}")

# Test nutrition database
if st.button("Test Nutrition Database"):
    with st.spinner("Testing nutrition database..."):
        try:
            current_url = FitnessAPI.get_api_url()
            test_query = {
                "query": "chicken breast",
                "limit": 3,
                "similarity_threshold": 0.6,
            }
            response = requests.post(
                f"{current_url}/v1/nutrition_search/search_nutrition_semantic/",
                json=test_query,
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Nutrition database working!")
                st.write(f"Found {result.get('results_found', 0)} results")
                if result.get("results"):
                    st.json(result["results"][0])  # Show first result
            else:
                st.error(f"❌ Nutrition database failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Nutrition database error: {str(e)}")

# Environment info
st.subheader("🌍 Environment Information")
env_info = {
    "Current API URL": FitnessAPI.get_api_url(),
    "Default API URL": API_BASE_URL,
    "Backend Host (env)": os.getenv("BACKEND_HOST", "Not set"),
    "Environment": os.getenv("ENVIRONMENT", "Not set"),
    "Python Version": "3.x",
    "Streamlit Version": (
        st.__version__ if hasattr(st, "__version__") else "Unknown"
    ),
}

for key, value in env_info.items():
    st.write(f"**{key}:** {value}")