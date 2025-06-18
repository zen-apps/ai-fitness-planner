import streamlit as st
from datetime import datetime
from utils.api_client import FitnessAPI, init_session_state, setup_api_settings_sidebar

# Configure page
st.set_page_config(
    page_title="Profile Setup - AI Fitness Planner",
    page_icon="👤",
    layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

st.header("👤 User Profile Setup")
st.markdown("Let's create your personalized fitness profile!")

# Test API connection first
current_api_url = FitnessAPI.get_api_url()
connection_test = FitnessAPI.test_connection()
if not connection_test["success"]:
    st.error(f"❌ Cannot connect to API at {current_api_url}")
    st.write(f"**Error:** {connection_test.get('error', 'Unknown error')}")
    st.write(f"**Attempted URL:** {connection_test.get('url', 'N/A')}")

    st.markdown(
        """
    **Troubleshooting:**
    1. Make sure the FastAPI server is running on the correct port
    2. Check if the API URL is correct (use the API Settings in sidebar)
    3. Verify network connectivity
    4. Try the API Testing page for more details
    """
    )

    if st.button("🔄 Retry Connection"):
        st.rerun()
    st.stop()
else:
    st.success(f"✅ Connected to API at {current_api_url}")

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