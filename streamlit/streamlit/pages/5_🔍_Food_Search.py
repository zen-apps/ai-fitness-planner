import streamlit as st
from utils.api_client import FitnessAPI, init_session_state, setup_api_settings_sidebar

# Configure page
st.set_page_config(
    page_title="Food Search - AI Fitness Planner",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

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