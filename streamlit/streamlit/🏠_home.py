import streamlit as st
from utils.api_client import init_session_state, setup_api_settings_sidebar

# Configure Streamlit page
st.set_page_config(
    page_title="AI Fitness Planner",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state and setup sidebar
init_session_state()
setup_api_settings_sidebar()

# Main content
st.title("🏋️‍♂️ AI Fitness Planner")
st.markdown("### Personalized Meal & Workout Plans Powered by LangChain Agents")

st.markdown(
    """
This application uses **LangChain Agents** to create personalized fitness and nutrition plans tailored specifically for you.

### 🤖 How It Works:

1. **Profile Manager Agent** - Analyzes your goals, body metrics, and preferences
2. **Meal Planner Agent** - Creates detailed nutrition plans using our USDA food database  
3. **Workout Planner Agent** - Designs comprehensive training programs with structured exercises
4. **Summary Agent** - Combines everything into actionable guidance with motivational insights

### 🎯 Features:
- **Personalized Macro Calculations** based on your goals (cut/bulk/maintenance)
- **Smart Food Recommendations** from curated USDA branded foods with vector search
- **Structured Meal Plans** with detailed foods, portions, and daily macro tracking
- **Complete Workout Programs** with exercises, sets, reps, and progression strategies
- **Real-time Plan Generation** using GPT-4o-mini and GPT-o3-mini powered agents with structured output

### 🚀 Get Started:
1. Set up your profile using the sidebar navigation
2. Generate your complete fitness plan
3. Follow your personalized recommendations

---
**Powered by:** FastAPI + MongoDB + LangChain + Streamlit
"""
)

# Quick stats from the database
st.subheader("📈 Platform Stats")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Available Foods", "5k", "Demo Dataset")

with col2:
    st.metric("Vector Search", "✅", "Enabled")

with col3:
    st.metric("Dietary Preferences", "6+", "Vegetarian, Vegan, Keto, etc.")

with col4:
    st.metric(
        "Workout Equipment Options", "6+", "Dumbbells, Barbells, Bodyweight, etc."
    )

with col5:
    st.metric("AI Agents", "4", "Structured Output")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Built by Josh Janzen |  All rights reserved.</p>
</div>
""",
    unsafe_allow_html=True,
)
