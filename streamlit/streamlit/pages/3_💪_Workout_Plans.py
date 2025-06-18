import streamlit as st
from utils.api_client import init_session_state, setup_api_settings_sidebar

# Configure page
st.set_page_config(
    page_title="Workout Plans - AI Fitness Planner",
    page_icon="💪",
    layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

st.header("💪 Workout Plans")

if not st.session_state.get("profile_created") and not st.session_state.get(
    "current_profile"
):
    st.warning("⚠️ Please create your profile first!")
    st.stop()

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