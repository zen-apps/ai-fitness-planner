import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
import os
from helpers.shiny_proxy_tracker import track_app_start
from sqlalchemy import text

# Page configuration
st.set_page_config(
    page_title="Data Science App Demos",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main():
    # Track user activity when they access the home page
    track_app_start(app_name="Home Page")

    # Sidebar data retention notice
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            """
            **📋 Data Usage Notice**
            
            Please note that all data entered into these demo applications 
            will be retained for demonstration and analytics purposes. 
            Do not enter sensitive or personal information.
            
            These demos are for testing and evaluation only.
            """
        )
    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown(
        '<h1 class="main-header">🚀 AI Fitness Planner</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Explore cutting-edge demo AI and Data Science applications</p>',
        unsafe_allow_html=True,
    )

    # Create demo cards
    demo_apps = [
        {
            "title": "🏭 Competitive Researcher",
            "page": "pages/00🏭_competitive_researcher.py",
            "description": "Intelligent competitive analysis and market research tool powered by AI and web search.",
            "features": [
                "Web search integration",
                "Automated competitive intelligence",
                "Market analysis with visualizations",
                "PDF document processing and analysis",
                "Structured data extraction",
            ],
            "status": "🟢 Active",
        },
        {
            "title": "🚀 CRM Intelligence",
            "page": "pages/01🚀_crm_intelligence.py",
            "description": "Advanced CRM data analysis and insights generation platform.",
            "features": [
                "Salesforce API integration",
                "Pipeline analysis and visualization",
                "Contact insights generation",
                "Sales script creation with AI",
                "Audio generation for sales calls",
            ],
            "status": "🟢 Active",
        },
        {
            "title": "📊 Financial Health Analyzer",
            "page": "pages/04📊_financial_health_analyzer.py",
            "description": "Comprehensive financial document analysis and health assessment tool.",
            "features": [
                "PDF financial document processing",
                "AI-powered financial analysis",
                "Health score calculations",
                "Interactive visualizations",
                "Risk assessment reporting",
            ],
            "status": "🟢 Active",
        },
        {
            "title": "💼 Customer Feedback Insights",
            "page": "pages/05💼_customer_feedback_insights.py",
            "description": "Advanced customer feedback analysis and sentiment insights platform.",
            "features": [
                "Multi-source feedback aggregation",
                "Sentiment analysis and scoring",
                "Trend identification and reporting",
                "Actionable insights generation",
                "Interactive dashboards",
            ],
            "status": "🟢 Active",
        },
        {
            "title": "📄 PDF Data Extractor",
            "page": "pages/06📄_pdf_data_extractor.py",
            "description": "AI-powered tool that extracts tabular data, text, and key-value pairs from PDFs and converts them into structured CSV files.",
            "features": [
                "PDF upload with drag-and-drop interface",
                "Table extraction from complex documents",
                "Key-value pair extraction for forms",
                "Column renaming and data cleanup",
                "CSV export functionality",
            ],
            "status": "🟢 Active",
        },
    ]

    # Display demo cards in a grid
    for i in range(0, len(demo_apps), 2):
        col1, col2 = st.columns(2)

        with col1:
            app = demo_apps[i]
            with st.container(border=True):
                st.page_link(app["page"], label=app["title"])
                st.write(app["description"])
                st.write(f"**Status:** {app['status']}")

                with st.expander("Key Features"):
                    for feature in app["features"]:
                        st.write(f"• {feature}")

                if st.button("Launch App", key=f"btn_{i}", use_container_width=True):
                    st.switch_page(app["page"])

        if i + 1 < len(demo_apps):
            with col2:
                app = demo_apps[i + 1]
                with st.container(border=True):
                    st.page_link(app["page"], label=app["title"])
                    st.write(app["description"])
                    st.write(f"**Status:** {app['status']}")

                    with st.expander("Key Features"):
                        for feature in app["features"]:
                            st.write(f"• {feature}")

                    if st.button(
                        "Launch App", key=f"btn_{i+1}", use_container_width=True
                    ):
                        st.switch_page(app["page"])

    # Statistics section
    st.markdown("---")
    st.markdown("## 📈 Demo Statistics")

    # Sample statistics (you can replace with real data)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Demos", value="5", delta="1 new this month")

    with col2:
        st.metric(label="Technologies Used", value="12+", delta="3 added recently")

    with col3:
        st.metric(label="Active Status", value="50%", delta="25% increase")

    with col4:
        st.metric(
            label="Last Updated",
            value=datetime.now().strftime("%B %Y"),
            delta="Current",
        )

    # Admin Section - Only visible to joshjanzen@gmail.com
    shinyproxy_username = os.getenv("SHINYPROXY_USERNAME", "")
    if shinyproxy_username == "joshjanzen@gmail.com":
        st.markdown("---")
        st.markdown("## 🔐 Admin Dashboard - User Activity Analytics")
        hide_joshjanzen = st.checkbox(
            "Hide Josh Janzen's Activity",
            value=True,
            help="Check this to hide your own activity from the dashboard.",
        )
        try:
            from helpers.shiny_proxy_tracker import tracker
            import plotly.express as px
            import plotly.graph_objects as go

            # Query all activity data
            if hide_joshjanzen:
                query = (
                    "SELECT * FROM shinyproxy_user_activity "
                    "WHERE username != 'joshjanzen@gmail.com' ORDER BY event_timestamp DESC"
                )
            else:
                query = "SELECT * FROM shinyproxy_user_activity ORDER BY event_timestamp DESC"
            df = pd.read_sql(query, con=tracker.engine)

            if not df.empty:
                st.write(f"**Total Records:** {len(df)}")

                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    unique_users = df["username"].nunique()
                    st.metric("Unique Users", unique_users)

                with col2:
                    total_sessions = df[df["event_type"] == "app_start"].shape[0]
                    st.metric("Total Sessions", total_sessions)

                with col3:
                    unique_apps = df["app_name"].nunique()
                    st.metric("Apps Accessed", unique_apps)

                with col4:
                    latest_activity = df["event_timestamp"].max()
                    if pd.notna(latest_activity):
                        st.metric(
                            "Last Activity", latest_activity.strftime("%m/%d %H:%M")
                        )

                # Activity by Event Type
                st.subheader("📊 Activity by Event Type")
                event_counts = df["event_type"].value_counts()
                fig_events = px.pie(
                    values=event_counts.values,
                    names=event_counts.index,
                    title="Distribution of Event Types",
                )
                st.plotly_chart(fig_events, use_container_width=True)

                # Activity Timeline
                st.subheader("📈 Activity Timeline")
                df["date"] = pd.to_datetime(df["event_timestamp"]).dt.date
                daily_activity = df.groupby("date").size().reset_index(name="count")
                fig_timeline = px.line(
                    daily_activity, x="date", y="count", title="Daily Activity Count"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

                # Activity by App
                if df["app_name"].notna().any():
                    st.subheader("🏭 Activity by Application")
                    app_counts = df.dropna(subset=["app_name"])[
                        "app_name"
                    ].value_counts()
                    fig_apps = px.bar(
                        x=app_counts.index,
                        y=app_counts.values,
                        title="Sessions by Application",
                        labels={"x": "Application", "y": "Number of Sessions"},
                    )
                    st.plotly_chart(fig_apps, use_container_width=True)

                # Recent Activity Table
                st.subheader("📋 Recent Activity")
                recent_df = df.head(10)[
                    [
                        "username",
                        "app_name",
                        "event_type",
                        "event_timestamp",
                        "ip_address",
                    ]
                ]
                st.dataframe(recent_df, use_container_width=True)

                # Full Data Download
                st.subheader("💾 Export Data")
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download Full Dataset as CSV",
                    data=csv_data,
                    file_name=f"shinyproxy_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            else:
                st.info("No activity data found in the database yet.")

        except Exception as e:
            st.error(f"Error loading admin dashboard: {e}")
            import traceback

            st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built by Josh Janzen | © 2025 Zen Software, LLC.  All rights reserved.</p>
        <p><em>Select a demo from the sidebar to get started!</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
