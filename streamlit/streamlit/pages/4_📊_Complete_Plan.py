import streamlit as st
from utils.api_client import FitnessAPI, init_session_state, setup_api_settings_sidebar

# Configure page
st.set_page_config(
    page_title="Complete Plan - AI Fitness Planner", page_icon="📊", layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

st.header("📊 Complete Fitness Plan")

if not st.session_state.get("current_profile") and not FitnessAPI.get_profile(
    st.session_state.user_id
):
    st.warning("⚠️ Please create your profile first!")
    if st.button("👤 Go to Profile Setup"):
        st.switch_page("pages/1_👤_Profile_Setup.py")
    st.stop()

st.markdown(
    "Generate your complete personalized fitness plan including both meal and workout plans using our advanced LangGraph workflow!"
)

# Plan configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍽️ Meal Plan Settings")
    st.info("Meal plan will be generated for 7 days")

with col2:
    st.subheader("💪 Workout Settings")
    workout_days_per_week = st.selectbox("Workout Days per Week", [2, 3, 4, 5], index=1)

# Generate button
if st.button("🚀 Generate Complete Plan", use_container_width=True, type="primary"):
    with st.spinner("🤖 LangGraph workflow is orchestrating your plan..."):
        # Show progress steps
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Initializing workflow...")
        progress_bar.progress(20)

        result = FitnessAPI.generate_langgraph_plan(
            st.session_state.user_id, workout_days_per_week
        )

        if result:
            status_text.text("✅ Workflow completed!")
            progress_bar.progress(100)

    if result:
        st.success("✅ Your complete fitness plan is ready!")

        # Show workflow execution steps
        if result.get("execution_steps"):
            with st.expander("🔍 Workflow Execution Steps", expanded=False):
                for i, step in enumerate(result["execution_steps"], 1):
                    st.write(f"{i}. {step}")

        # Show any errors
        if result.get("errors"):
            st.warning("⚠️ Some issues occurred during generation:")
            for error in result["errors"]:
                st.write(f"- {error}")

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
                with st.expander("📋 Detailed Meal Plan", expanded=False):
                    st.markdown(meal_plan["plan_content"])

        # Workout Plan Section
        if result.get("workout_plan"):
            st.subheader("💪 Workout Plan")
            workout_plan = result["workout_plan"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Training Style", workout_plan.get("training_style", "N/A"))
            with col2:
                st.metric("Split Type", workout_plan.get("split_type", "N/A"))
            with col3:
                st.metric("Days/Week", workout_plan.get("days_per_week", "N/A"))
            with col4:
                duration = workout_plan.get("duration_minutes", "N/A")
                st.metric("Duration", f"{duration} min" if duration != "N/A" else "N/A")

            if workout_plan.get("plan_content"):
                with st.expander("🏋️‍♂️ Detailed Workout Plan", expanded=False):
                    st.markdown(workout_plan["plan_content"])

        # Plan metadata
        if result.get("generated_at"):
            st.caption(f"Plan generated on: {result['generated_at']}")

        st.balloons()
