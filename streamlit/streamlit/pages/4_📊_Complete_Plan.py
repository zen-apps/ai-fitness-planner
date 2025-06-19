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
    "Generate your complete personalized fitness plan including both meal and workout plans!"
)

# Plan configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍽️ Meal Plan Settings")
    meal_plan_days = st.selectbox("Meal Plan Days Out", [3, 7, 14], index=1)

with col2:
    st.subheader("💪 Workout Settings")
    workout_days_per_week = st.selectbox("Workout Days per Week", [2, 3, 4, 5], index=1)

# Plan generation options
st.subheader("🤖 AI Generation Method")
generation_method = st.radio(
    "Choose generation method:",
    ["🔗 LangGraph Workflow (Recommended)", "🧠 Individual Agents"],
    help="LangGraph provides better coordination between agents",
)

# Generate button
if st.button("🚀 Generate Complete Plan", use_container_width=True, type="primary"):
    if generation_method.startswith("🔗"):
        with st.spinner("🤖 LangGraph workflow is orchestrating your plan..."):
            # Show progress steps
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 Initializing workflow...")
            progress_bar.progress(20)

            result = FitnessAPI.generate_langgraph_plan(
                st.session_state.user_id, meal_plan_days, workout_days_per_week
            )

            if result:
                status_text.text("✅ Workflow completed!")
                progress_bar.progress(100)
    else:
        with st.spinner("🤖 AI Agents are creating your personalized plan..."):
            result = FitnessAPI.generate_complete_plan(
                st.session_state.user_id, meal_plan_days, workout_days_per_week
            )

    if result:
        st.success("✅ Your complete fitness plan is ready!")

        # Show workflow execution steps for LangGraph
        if generation_method.startswith("🔗") and result.get("execution_steps"):
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
                with st.expander("📋 Detailed Meal Plan", expanded=True):
                    # Try to parse structured content, fallback to markdown
                    try:
                        import json
                        import re
                        
                        plan_content = meal_plan["plan_content"]
                        
                        # Try to extract JSON from the content
                        json_match = re.search(r'\{.*\}', plan_content, re.DOTALL)
                        if json_match:
                            plan_data = json.loads(json_match.group())
                            
                            if "daily_plans" in plan_data:
                                # Create tabs for each day
                                days = plan_data["daily_plans"]
                                if days:
                                    day_tabs = st.tabs([f"Day {day.get('day', i+1)}" for i, day in enumerate(days)])
                                    
                                    for tab, day_plan in zip(day_tabs, days):
                                        with tab:
                                            if "meals" in day_plan:
                                                for meal in day_plan["meals"]:
                                                    meal_type = meal.get("meal_type", "Meal")
                                                    st.markdown(f"### {meal_type}")
                                                    
                                                    if "foods" in meal:
                                                        for food in meal["foods"]:
                                                            food_name = food.get("food", "Unknown Food")
                                                            portion = food.get("portion", "1 serving")
                                                            
                                                            col1, col2 = st.columns([3, 1])
                                                            with col1:
                                                                st.write(f"• **{food_name}**")
                                                            with col2:
                                                                st.write(f"`{portion}`")
                                                    
                                                    # Show meal macros if available
                                                    if "macros" in meal:
                                                        meal_macros = meal["macros"]
                                                        cols = st.columns(4)
                                                        with cols[0]:
                                                            st.metric("Cal", f"{meal_macros.get('calories', 0)}")
                                                        with cols[1]:
                                                            st.metric("P", f"{meal_macros.get('protein_g', 0)}g")
                                                        with cols[2]:
                                                            st.metric("C", f"{meal_macros.get('carbs_g', 0)}g")
                                                        with cols[3]:
                                                            st.metric("F", f"{meal_macros.get('fat_g', 0)}g")
                                                    
                                                    st.divider()
                            else:
                                # Fallback to markdown if structure is different
                                st.markdown(plan_content)
                        else:
                            # No JSON found, display as markdown
                            st.markdown(plan_content)
                            
                    except (json.JSONDecodeError, Exception):
                        # Fallback to markdown display
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
                with st.expander("🏋️‍♂️ Detailed Workout Plan", expanded=True):
                    # Try to parse structured content, fallback to markdown
                    try:
                        import json
                        import re
                        
                        plan_content = workout_plan["plan_content"]
                        
                        # Try to extract JSON from the content
                        json_match = re.search(r'\{.*\}', plan_content, re.DOTALL)
                        if json_match:
                            plan_data = json.loads(json_match.group())
                            
                            if "weekly_schedule" in plan_data:
                                # Create tabs for each workout day
                                workouts = plan_data["weekly_schedule"]
                                if workouts:
                                    workout_tabs = st.tabs([f"{workout.get('day', f'Day {i+1}')}" for i, workout in enumerate(workouts)])
                                    
                                    for tab, workout_day in zip(workout_tabs, workouts):
                                        with tab:
                                            # Show workout info
                                            if workout_day.get("focus"):
                                                st.markdown(f"**Focus:** {workout_day['focus']}")
                                            if workout_day.get("duration_minutes"):
                                                st.markdown(f"**Duration:** {workout_day['duration_minutes']} minutes")
                                            
                                            st.divider()
                                            
                                            # Show exercises
                                            if "exercises" in workout_day:
                                                for i, exercise in enumerate(workout_day["exercises"], 1):
                                                    exercise_name = exercise.get("exercise", "Unknown Exercise")
                                                    
                                                    col1, col2 = st.columns([3, 2])
                                                    with col1:
                                                        st.markdown(f"**{i}. {exercise_name}**")
                                                        if exercise.get("muscle_group"):
                                                            st.caption(f"Target: {exercise['muscle_group']}")
                                                    
                                                    with col2:
                                                        # Exercise details in a compact format
                                                        details = []
                                                        if exercise.get("sets"):
                                                            details.append(f"{exercise['sets']} sets")
                                                        if exercise.get("reps"):
                                                            details.append(f"{exercise['reps']} reps")
                                                        if exercise.get("weight"):
                                                            details.append(f"{exercise['weight']}")
                                                        if exercise.get("rest_seconds"):
                                                            rest_min = exercise["rest_seconds"] // 60
                                                            rest_sec = exercise["rest_seconds"] % 60
                                                            if rest_min > 0:
                                                                details.append(f"{rest_min}:{rest_sec:02d} rest")
                                                            else:
                                                                details.append(f"{rest_sec}s rest")
                                                        
                                                        if details:
                                                            st.write(" • ".join(details))
                                                    
                                                    # Exercise notes
                                                    if exercise.get("notes"):
                                                        st.caption(f"💡 {exercise['notes']}")
                                                    
                                                    if i < len(workout_day["exercises"]):
                                                        st.divider()
                                            
                                            # Show workout notes
                                            if workout_day.get("notes"):
                                                st.info(f"📝 **Workout Notes:** {workout_day['notes']}")
                            else:
                                # Fallback to markdown if structure is different
                                st.markdown(plan_content)
                        else:
                            # No JSON found, display as markdown
                            st.markdown(plan_content)
                            
                    except (json.JSONDecodeError, Exception):
                        # Fallback to markdown display
                        st.markdown(workout_plan["plan_content"])

        # Plan metadata
        if result.get("generated_at"):
            st.caption(f"Plan generated on: {result['generated_at']}")

        st.balloons()
