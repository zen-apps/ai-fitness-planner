import streamlit as st
from datetime import datetime

def render_footer():
    """Render the application footer with attribution"""
    
    # Add some spacing
    st.markdown("---")
    
    # Attribution section
    st.markdown(
        """
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 10px;">
            <hr style="margin: 20px 0; border-color: #dee2e6;">
            <div style="text-align: center;">
                <p style="color: #495057; margin-bottom: 10px; font-size: 16px;">
                    🚀 <strong>Built by Josh Janzen</strong> 
                </p>
                <p style="color: #6c757d; font-size: 14px; margin-bottom: 10px;">
                    AI Fitness Planner - Demonstrating LangChain Agents & Modern AI Architecture
                </p>
                <p style="color: #6c757d; font-size: 12px; margin-bottom: 0;">
                    Powered by: FastAPI • MongoDB • LangChain • OpenAI • Streamlit<br>
                    © {year} Josh Janzen. Demo application - not for commercial use.
                </p>
            </div>
        </div>
        """.format(year=datetime.now().year),
        unsafe_allow_html=True,
    )