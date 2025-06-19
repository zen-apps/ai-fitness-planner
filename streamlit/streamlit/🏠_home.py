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
## Welcome to Your AI Fitness Journey! 🚀

This application uses **LangChain Agents** to create personalized fitness and nutrition plans tailored specifically for you.

### 🤖 How It Works:

1. **Profile Manager Agent** - Analyzes your goals, body metrics, and preferences
2. **Meal Planner Agent** - Creates nutrition plans using our USDA food database
3. **Workout Planner Agent** - Designs training programs based on your goals
4. **Summary Agent** - Combines everything into actionable guidance

### 🎯 Features:
- **Personalized Macro Calculations** based on your goals (cut/bulk/maintenance)
- **Smart Food Recommendations** from 300k+ USDA branded foods
- **Adaptive Workout Plans** for strength, hypertrophy, or endurance
- **Real-time Plan Generation** using GPT-4 powered agents

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

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Available Foods", "300k+", "USDA Database")

with col2:
    st.metric("Workout Styles", "3+", "Strength, Hypertrophy, Endurance")

with col3:
    st.metric("Plan Types", "5+", "Cut, Bulk, Maintenance, Recomp")

with col4:
    st.metric("AI Agents", "4", "LangChain Powered")

# Navigation help
st.markdown(
    """
---
### 📋 Navigation Guide:
Use the sidebar to access different sections:
- **👤 Profile Setup** - Create and manage your fitness profile
- **📊 Complete Plan** - Get your full fitness plan with LangGraph
- **🔍 Food Search** - Search the USDA nutrition database
- **🧪 API Testing** - Test and debug API connections
"""
)
