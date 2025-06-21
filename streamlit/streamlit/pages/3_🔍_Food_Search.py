import streamlit as st
import requests
from utils.api_client import FitnessAPI, init_session_state, setup_api_settings_sidebar

# Configure page
st.set_page_config(
    page_title="Food Search - AI Fitness Planner", page_icon="🔍", layout="wide"
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

st.header("🔍 Food Database Search")
st.markdown(
    "Search through our comprehensive USDA nutrition database with advanced filtering options."
)


def display_search_results(results):
    """Display basic search results"""
    for i, food in enumerate(results):
        with st.expander(
            f"🥘 {food.get('description', 'Unknown Food')} - {food.get('brand_owner', 'Unknown Brand')}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Product Info:**")
                st.write(f"**Brand:** {food.get('brand_name', 'N/A')}")
                st.write(f"**Category:** {food.get('food_category', 'N/A')}")
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

                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Calories", f"{per_100g.get('energy_kcal', 0)} kcal")
                        st.metric("Protein", f"{per_100g.get('protein_g', 0)} g")
                    with metrics_col2:
                        st.metric("Carbs", f"{per_100g.get('carbs_g', 0)} g")
                        st.metric("Fat", f"{per_100g.get('total_fat_g', 0)} g")

                    macro_breakdown = nutrition.get("macro_breakdown", {})
                    if macro_breakdown.get("primary_macro_category"):
                        st.write(
                            f"**Primary Macro:** {macro_breakdown['primary_macro_category'].replace('_', ' ').title()}"
                        )
                else:
                    st.warning("Nutrition data not available")


def display_semantic_results(results):
    """Display semantic search results with similarity scores"""
    for i, food in enumerate(results):
        similarity_score = food.get("similarity_score", 0)
        score_color = (
            "🟢" if similarity_score > 0.8 else "🟡" if similarity_score > 0.7 else "🟠"
        )

        with st.expander(
            f"{score_color} {food.get('description', 'Unknown Food')} - {food.get('brand_owner', 'Unknown Brand')} (Match: {similarity_score:.1%})"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Product Info:**")
                st.write(f"**Brand:** {food.get('brand_name', 'N/A')}")
                st.write(f"**Category:** {food.get('food_category', 'N/A')}")
                st.write(f"**Serving Size:** {food.get('serving_size', 0)} g")

                # Show what matched - fixed: no nested expanders
                if food.get("matched_content"):
                    st.markdown("**📄 AI Match Context:**")
                    with st.container():
                        st.caption(food["matched_content"])

            with col2:
                nutrition = food.get("nutrition_per_100g", {})

                st.markdown("**Nutrition per 100g:**")

                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Calories", f"{nutrition.get('calories', 0):.0f} kcal")
                    st.metric("Protein", f"{nutrition.get('protein_g', 0):.1f} g")
                with metrics_col2:
                    st.metric("Carbs", f"{nutrition.get('carbs_g', 0):.1f} g")
                    st.metric("Fat", f"{nutrition.get('fat_g', 0):.1f} g")

                # Additional info
                primary_macro = food.get("primary_macro_category", "unknown")
                if primary_macro != "unknown":
                    st.write(
                        f"**Primary Macro:** {primary_macro.replace('_', ' ').title()}"
                    )

                if food.get("is_high_protein"):
                    st.info("💪 High Protein Food")


def display_hybrid_results(results):
    """Display hybrid search results with scoring breakdown"""
    for i, food in enumerate(results):
        hybrid_score = food.get("hybrid_score", 0)
        semantic_score = food.get("semantic_score", 0)
        traditional_score = food.get("traditional_score", 0)

        score_icon = (
            "🥇" if hybrid_score > 0.8 else "🥈" if hybrid_score > 0.6 else "🥉"
        )

        with st.expander(
            f"{score_icon} {food.get('description', 'Unknown Food')} - {food.get('brand_owner', 'Unknown Brand')} (Score: {hybrid_score:.2f})"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Product Info:**")
                st.write(f"**Brand:** {food.get('brand_name', 'N/A')}")
                st.write(f"**Category:** {food.get('food_category', 'N/A')}")
                st.write(f"**Serving Size:** {food.get('serving_size', 0)} g")

                # Show scoring breakdown - improved layout
                st.markdown("**Search Scores:**")
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.metric("🧠 AI Score", f"{semantic_score:.2f}")
                    st.metric("📝 Text Score", f"{traditional_score:.2f}")
                with score_col2:
                    st.metric("🎯 Combined", f"{hybrid_score:.2f}")

            with col2:
                nutrition = food.get("nutrition_per_100g", {})

                st.markdown("**Nutrition per 100g:**")

                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Calories", f"{nutrition.get('calories', 0):.0f} kcal")
                    st.metric("Protein", f"{nutrition.get('protein_g', 0):.1f} g")
                with metrics_col2:
                    st.metric("Carbs", f"{nutrition.get('carbs_g', 0):.1f} g")
                    st.metric("Fat", f"{nutrition.get('fat_g', 0):.1f} g")

                # Additional info
                primary_macro = food.get("primary_macro_category", "unknown")
                if primary_macro != "unknown":
                    st.write(
                        f"**Primary Macro:** {primary_macro.replace('_', ' ').title()}"
                    )

                if food.get("is_high_protein"):
                    st.info("💪 High Protein Food")

                nutrition_density = food.get("nutrition_density_score", 0)
                if nutrition_density > 0:
                    st.metric("🎯 Nutrition Density", f"{nutrition_density:.1f}")


# Database selection
# st.subheader("📊 Database Settings")
# use_full_database = st.checkbox(
#     "Use full USDA database",
#     value=False,
#     help="Use the complete USDA database vs the sample dataset. Requires full database to be imported."
# )
use_full_database = False
# Database availability check
if use_full_database:
    try:
        db_status = FitnessAPI.check_database_availability()
        if not db_status.get("full_database", {}).get("available", False):
            st.error(
                "❌ Full USDA database is not available. Only sampled USDA data is available. "
                "Please import the full database first or uncheck 'Use full USDA database'."
            )
            use_full_database = False  # Override to use sample
        else:
            st.success(
                f"✅ Full database available with {db_status['full_database']['document_count']:,} foods"
            )
    except Exception as e:
        st.warning(f"⚠️ Could not check database availability: {str(e)}")
        use_full_database = False
else:
    try:
        db_status = FitnessAPI.check_database_availability()
        sample_count = db_status.get("sample_database", {}).get("document_count", 0)
        if sample_count > 0:
            st.info(f"📋 Using sample database with {sample_count:,} foods")
    except:
        pass

# Search method selection
search_method = st.radio(
    "Search Method:",
    ["🔍 Basic Search", "🧠 Semantic Search", "🎯 Advanced Filters"],
    horizontal=True,
    help="Basic: Simple name/brand search. Semantic: AI-powered natural language search. Advanced: Nutrition-based filtering.",
)

if search_method == "🔍 Basic Search":
    # Basic search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search for foods:",
            placeholder="e.g., chicken breast, greek yogurt, quinoa...",
            key="basic_query",
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
                display_search_results(results["results"])
            else:
                st.warning("No results found. Try a different search term!")

elif search_method == "🧠 Semantic Search":
    # Semantic search interface
    st.markdown("### 🧠 AI-Powered Semantic Search")
    st.info(
        "Use natural language to describe what you're looking for (e.g., 'high protein breakfast food' or 'low carb snack for keto diet')"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        semantic_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g., high protein low carb breakfast, post-workout recovery food...",
            key="semantic_query",
        )

    with col2:
        similarity_threshold = st.slider(
            "Match Quality",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values = more precise matches",
        )

    with col3:
        semantic_limit = st.selectbox(
            "Results", [5, 10, 15, 20], index=1, key="semantic_limit"
        )

    # Dietary restrictions
    with st.expander("🥗 Dietary Restrictions (Optional)"):
        col1, col2, col3 = st.columns(3)

        with col1:
            vegan = st.checkbox("Vegan")
        with col2:
            vegetarian = st.checkbox("Vegetarian")
        with col3:
            gluten_free = st.checkbox("Gluten-Free")

    if st.button("🧠 Semantic Search", use_container_width=True) and semantic_query:
        # Build dietary restrictions list
        restrictions = []
        if vegan:
            restrictions.append("vegan")
        if vegetarian:
            restrictions.append("vegetarian")
        if gluten_free:
            restrictions.append("gluten-free")

        with st.spinner("Performing AI-powered search..."):
            try:
                api_url = FitnessAPI.get_api_url()
                request_data = {
                    "query": semantic_query,
                    "dietary_restrictions": restrictions,
                    "macro_goals": {},
                    "limit": semantic_limit,
                    "similarity_threshold": similarity_threshold,
                    "use_full_database": use_full_database,
                }

                response = requests.post(
                    f"{api_url}/v1/nutrition_search/search_nutrition_semantic/",
                    json=request_data,
                    timeout=30,
                )

                if response.status_code == 200:
                    results = response.json()
                    st.success(
                        f"Found {results['results_found']} semantic matches in {results['search_time_ms']}ms"
                    )
                    display_semantic_results(results["results"])
                else:
                    st.error(f"Search failed: {response.text}")

            except Exception as e:
                st.error(f"Error performing semantic search: {str(e)}")

elif search_method == "🎯 Advanced Filters":
    # Advanced nutrition-based search
    st.markdown("### 🎯 Nutrition-Based Search")
    st.info("Search for foods that meet specific nutritional criteria")

    col1, col2 = st.columns([2, 1])

    with col1:
        advanced_query = st.text_input(
            "Food name or description:",
            placeholder="e.g., protein powder, chicken, oats...",
            key="advanced_query",
        )

    with col2:
        hybrid_limit = st.selectbox(
            "Results", [5, 10, 15, 20], index=1, key="hybrid_limit"
        )

    # Nutrition filters
    st.markdown("#### 🥇 Nutrition Targets (per 100g)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        protein_min = st.number_input(
            "Min Protein (g)", min_value=0.0, max_value=100.0, value=0.0, step=1.0
        )

    with col2:
        carbs_max = st.number_input(
            "Max Carbs (g)", min_value=0.0, max_value=100.0, value=100.0, step=1.0
        )

    with col3:
        calories_max = st.number_input(
            "Max Calories", min_value=0, max_value=1000, value=1000, step=10
        )

    with col4:
        semantic_weight = st.slider(
            "AI vs Text Match",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0 = pure text search, 1 = pure AI search",
        )

    # Dietary restrictions for advanced search
    with st.expander("🥗 Dietary Restrictions (Optional)"):
        restrictions_text = st.text_input(
            "Dietary restrictions (comma-separated):",
            placeholder="e.g., vegan, gluten-free, dairy-free",
            help="Enter dietary restrictions separated by commas",
        )

    if st.button("🎯 Advanced Search", use_container_width=True) and advanced_query:
        with st.spinner("Performing advanced nutrition search..."):
            try:
                api_url = FitnessAPI.get_api_url()

                params = {
                    "query": advanced_query,
                    "dietary_restrictions": restrictions_text,
                    "protein_min": protein_min,
                    "carbs_max": carbs_max,
                    "calories_max": calories_max,
                    "limit": hybrid_limit,
                    "semantic_weight": semantic_weight,
                    "use_full_database": use_full_database,
                }

                response = requests.get(
                    f"{api_url}/v1/nutrition_search/search_nutrition_hybrid/",
                    params=params,
                    timeout=30,
                )

                if response.status_code == 200:
                    results = response.json()
                    st.success(
                        f"Found {results['results_found']} foods matching your criteria"
                    )

                    # Show search weights
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "AI Search Weight", f"{results['semantic_weight']:.0%}"
                        )
                    with col2:
                        st.metric(
                            "Text Search Weight", f"{results['traditional_weight']:.0%}"
                        )

                    display_hybrid_results(results["results"])
                else:
                    st.error(f"Search failed: {response.text}")

            except Exception as e:
                st.error(f"Error performing advanced search: {str(e)}")

# Add help section
with st.expander("❓ Search Help"):
    st.markdown(
        """
    ### Search Types:
    
    **🔍 Basic Search:** Traditional name/brand search through the database
    - Best for: Finding specific foods you know by name
    - Example: "greek yogurt", "chicken breast"
    
    **🧠 Semantic Search:** AI-powered natural language search
    - Best for: Describing what you want nutritionally
    - Example: "high protein breakfast food", "low carb keto snack"
    
    **🎯 Advanced Filters:** Combine AI search with specific nutrition criteria
    - Best for: Finding foods that meet exact macro goals
    - Example: "protein powder" with >20g protein, <5g carbs
    
    ### Tips:
    - Use specific terms for better results
    - Try different search methods if you don't find what you're looking for
    - Semantic search works best with descriptive queries
    - Advanced filters help narrow down large result sets
    """
    )
