import streamlit as st
from utils.api_client import init_session_state, setup_api_settings_sidebar

# Configure page
st.set_page_config(
    page_title="Meal Plans - AI Fitness Planner",
    page_icon="🍽️",
    layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

st.header("🍽️ Meal Plans")

if not st.session_state.get("profile_created") and not st.session_state.get(
    "current_profile"
):
    st.warning("⚠️ Please create your profile first!")
    st.stop()

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