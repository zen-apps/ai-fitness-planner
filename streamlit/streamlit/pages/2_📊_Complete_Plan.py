import streamlit as st
from utils.api_client import FitnessAPI, init_session_state
from utils.footer import render_footer

# Configure page
st.set_page_config(
    page_title="Complete Plan - AI Fitness Planner", page_icon="📊", layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()

# Add meal plan days selector in sidebar
with st.sidebar:
    st.subheader("🍽️ Meal Plan Settings")
    meal_plan_days = st.selectbox(
        "Number of days for meal plan",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=0,  # Default to 1 day
        help="Choose how many days of meal planning to generate. Fewer days = faster generation.",
    )

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
    st.info(
        f"Meal plan will be generated for {meal_plan_days} day{'s' if meal_plan_days > 1 else ''}"
    )

with col2:
    st.subheader("💪 Workout Settings")

    # Get current profile to display workout frequency
    current_profile = st.session_state.get("current_profile") or FitnessAPI.get_profile(
        st.session_state.user_id
    )
    if current_profile and current_profile.get("workout_frequency"):
        workout_freq = current_profile["workout_frequency"]
        st.info(f"Workout frequency: {workout_freq} days per week")
    else:
        st.info("Workout frequency is configured in your profile settings")

# AI Model Settings
st.sidebar.subheader("🤖 AI Model Settings")
use_o3_mini = st.sidebar.checkbox(
    "Use O3-mini Reasoning (takes longer, better results)", value=True
)

# with col2:
#     use_full_database = st.checkbox(
#         "Use full USDA database",
#         value=False,
#         help="Use the complete USDA database vs the sample dataset. Requires full database to be imported.",
#     )
use_full_database = False
# Database availability check
if use_full_database:
    # Check if full database is available
    try:
        db_status = FitnessAPI.check_database_availability()
        if not db_status.get("full_database", {}).get("available", False):
            st.error(
                "❌ Full USDA database is not available. Only sampled USDA data is available. "
                "Please import the full database first or uncheck 'Use full USDA database'."
            )
            use_full_database = False  # Override to use sample
    except Exception as e:
        st.warning(f"⚠️ Could not check database availability: {str(e)}")
        use_full_database = False

# Generate button
if st.button("🚀 Generate Complete Plan", use_container_width=True, type="primary"):
    with st.spinner("🤖 LangGraph workflow is orchestrating your plan..."):
        # Show progress steps
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Initializing workflow...")
        progress_bar.progress(20)

        result = FitnessAPI.generate_langgraph_plan(
            st.session_state.user_id, use_o3_mini, use_full_database, meal_plan_days
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

            # Display plan name and overview
            if meal_plan.get("plan_name"):
                st.markdown(f"**{meal_plan['plan_name']}**")

            if meal_plan.get("target_macros"):
                st.markdown("**Daily Targets:**")

                col1, col2, col3, col4 = st.columns(4)

                macros = meal_plan["target_macros"]
                with col1:
                    st.metric("Calories", f"{macros.get('calories', 0):,.0f}")
                with col2:
                    st.metric("Protein", f"{macros.get('protein_g', 0):.0f}g")
                with col3:
                    st.metric("Carbs", f"{macros.get('carbs_g', 0):.0f}g")
                with col4:
                    st.metric("Fat", f"{macros.get('fat_g', 0):.0f}g")

            # Display daily meal plans in structured format
            if meal_plan.get("daily_plans"):
                plan_days = len(meal_plan["daily_plans"])
                with st.expander(
                    f"📋 {plan_days}-Day Detailed Meal Plan", expanded=True
                ):
                    for day_plan in meal_plan["daily_plans"]:
                        day_name = (
                            day_plan.get("day_name")
                            or f"Day {day_plan.get('day', '?')}"
                        )
                        st.markdown(f"### {day_name}")

                        # Display meals for this day
                        for meal in day_plan.get("meals", []):
                            st.markdown(f"**{meal.get('meal_name', 'Meal')}**")

                            # Display foods in this meal
                            for food in meal.get("foods", []):
                                st.write(
                                    f"• {food.get('food_name', 'Food')} - {food.get('portion', 'N/A')} "
                                    f"({food.get('calories', 0):.0f} cal, "
                                    f"{food.get('protein_g', 0):.1f}g protein)"
                                )

                            # Display meal totals
                            meal_macros = meal.get("total_macros", {})
                            if meal_macros:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.caption(
                                        f"🔥 {meal_macros.get('calories', 0):.0f} cal"
                                    )
                                with col2:
                                    st.caption(
                                        f"🥩 {meal_macros.get('protein_g', 0):.1f}g"
                                    )
                                with col3:
                                    st.caption(
                                        f"🍞 {meal_macros.get('carbs_g', 0):.1f}g"
                                    )
                                with col4:
                                    st.caption(f"🥑 {meal_macros.get('fat_g', 0):.1f}g")

                            # Preparation notes
                            if meal.get("preparation_notes"):
                                st.caption(f"📝 {meal['preparation_notes']}")

                            st.markdown("---")

                        # Daily totals
                        daily_totals = day_plan.get("daily_totals", {})
                        if daily_totals:
                            st.markdown("**Daily Totals:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Calories", f"{daily_totals.get('calories', 0):.0f}"
                                )
                            with col2:
                                st.metric(
                                    "Protein",
                                    f"{daily_totals.get('protein_g', 0):.1f}g",
                                )
                            with col3:
                                st.metric(
                                    "Carbs", f"{daily_totals.get('carbs_g', 0):.1f}g"
                                )
                            with col4:
                                st.metric("Fat", f"{daily_totals.get('fat_g', 0):.1f}g")

                        st.markdown("---")

            # Display key principles and shopping tips
            col1, col2 = st.columns(2)

            with col1:
                if meal_plan.get("key_principles"):
                    st.markdown("**🎯 Key Principles:**")
                    for principle in meal_plan["key_principles"]:
                        st.write(f"• {principle}")

            with col2:
                if meal_plan.get("shopping_tips"):
                    st.markdown("**🛒 Shopping Tips:**")
                    for tip in meal_plan["shopping_tips"]:
                        st.write(f"• {tip}")

            # Show metadata
            if meal_plan.get("available_foods_count"):
                st.caption(
                    f"📊 Plan created using {meal_plan['available_foods_count']} available foods from database"
                )

        # Workout Plan Section
        if result.get("workout_plan"):
            st.subheader("💪 Workout Plan")
            workout_plan = result["workout_plan"]

            # Display plan name and overview
            if workout_plan.get("plan_name"):
                st.markdown(f"**{workout_plan['plan_name']}**")

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

            # Display weekly schedule in structured format
            if workout_plan.get("weekly_schedule"):
                with st.expander("🏋️‍♂️ Weekly Workout Schedule", expanded=True):
                    for workout_day in workout_plan["weekly_schedule"]:
                        day_name = (
                            workout_day.get("day_name")
                            or f"Day {workout_day.get('day', '?')}"
                        )
                        st.markdown(
                            f"### {day_name} - {workout_day.get('focus', 'Workout')}"
                        )

                        # Warm-up
                        if workout_day.get("warm_up"):
                            st.markdown("**🔥 Warm-up:**")
                            for warmup in workout_day["warm_up"]:
                                st.write(f"• {warmup}")

                        # Main exercises
                        if workout_day.get("exercises"):
                            st.markdown("**💪 Exercises:**")

                            # Create table-like display for exercises
                            ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(
                                [3, 1, 1, 2]
                            )

                            with ex_col1:
                                st.write("**Exercise**")
                            with ex_col2:
                                st.write("**Sets**")
                            with ex_col3:
                                st.write("**Reps**")
                            with ex_col4:
                                st.write("**Rest**")

                            st.markdown("---")

                            for exercise in workout_day["exercises"]:
                                ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(
                                    [3, 1, 1, 2]
                                )

                                with ex_col1:
                                    st.write(exercise.get("exercise_name", "Exercise"))
                                    if exercise.get("notes"):
                                        st.caption(f"💡 {exercise['notes']}")
                                with ex_col2:
                                    st.write(str(exercise.get("sets", "N/A")))
                                with ex_col3:
                                    st.write(str(exercise.get("reps", "N/A")))
                                with ex_col4:
                                    rest_time = exercise.get("rest_seconds", 0)
                                    if rest_time >= 60:
                                        st.write(
                                            f"{rest_time // 60}min {rest_time % 60}s"
                                        )
                                    else:
                                        st.write(f"{rest_time}s")

                        # Cool-down
                        if workout_day.get("cool_down"):
                            st.markdown("**🧘 Cool-down:**")
                            for cooldown in workout_day["cool_down"]:
                                st.write(f"• {cooldown}")

                        # Estimated duration
                        if workout_day.get("estimated_duration"):
                            st.caption(
                                f"⏱️ Estimated duration: {workout_day['estimated_duration']} minutes"
                            )

                        st.markdown("---")

            # Display additional workout plan info
            col1, col2 = st.columns(2)

            with col1:
                if workout_plan.get("key_principles"):
                    st.markdown("**🎯 Key Training Principles:**")
                    for principle in workout_plan["key_principles"]:
                        st.write(f"• {principle}")

                if workout_plan.get("progression_strategy"):
                    st.markdown("**📈 Progression Strategy:**")
                    st.write(workout_plan["progression_strategy"])

            with col2:
                if workout_plan.get("equipment_needed"):
                    st.markdown("**🛠️ Equipment Needed:**")
                    for equipment in workout_plan["equipment_needed"]:
                        st.write(f"• {equipment}")

        # Plan metadata
        if result.get("generated_at"):
            st.caption(f"Plan generated on: {result['generated_at']}")

        st.balloons()

# Footer
render_footer()
