import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict
import os
import PyPDF2
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from dotenv import load_dotenv

load_dotenv("./config/dev.env")


# Pydantic models for structured output
class PositioningScores(BaseModel):
    """Positioning scores for competitive analysis"""

    price: int = Field(description="Price competitiveness score (1-10)", ge=1, le=10)
    innovation: int = Field(description="Innovation score (1-10)", ge=1, le=10)
    market_share: int = Field(description="Market share score (1-10)", ge=1, le=10)
    customer_satisfaction: int = Field(
        description="Customer satisfaction score (1-10)", ge=1, le=10
    )


class CompanyAnalysis(BaseModel):
    """Analysis data for a single company"""

    name: str = Field(description="Company name")
    overview: str = Field(description="Brief company overview (max 100 words)")
    strengths: List[str] = Field(
        description="List of company strengths", min_items=2, max_items=3
    )
    weaknesses: List[str] = Field(
        description="List of company weaknesses", min_items=2, max_items=3
    )
    opportunities: List[str] = Field(
        description="List of opportunities", min_items=2, max_items=3
    )
    threats: List[str] = Field(description="List of threats", min_items=2, max_items=3)
    recent_news_sentiment: str = Field(
        description="Recent news sentiment", pattern="^(positive|neutral|negative)$"
    )
    sentiment_explanation: str = Field(
        description="Brief explanation of sentiment (max 50 words)"
    )
    tech_stack: List[str] = Field(
        description="Technology stack or key technologies", min_items=1, max_items=4
    )
    positioning_scores: PositioningScores = Field(
        description="Competitive positioning scores"
    )


class CompetitiveInsights(BaseModel):
    """Competitive insights and recommendations"""

    key_differentiators: List[str] = Field(
        description="Key differentiators between companies", min_items=2, max_items=3
    )
    market_gaps: List[str] = Field(
        description="Identified market gaps", min_items=1, max_items=2
    )
    strategic_recommendations: List[str] = Field(
        description="Strategic recommendations", min_items=3, max_items=4
    )


class ActionItem(BaseModel):
    """Individual action item with details"""

    title: str = Field(description="Action item title")
    description: str = Field(description="Detailed description of the action")
    priority: str = Field(description="Priority level", pattern="^(High|Medium|Low)$")
    timeline: str = Field(description="Expected timeline for completion")
    resources_needed: List[str] = Field(
        description="Resources or departments needed", max_items=3
    )
    expected_impact: str = Field(description="Expected business impact")


class ActionPlan(BaseModel):
    """Comprehensive action plan to beat the competition"""

    immediate_actions: List[ActionItem] = Field(
        description="Actions to take in the next 30-90 days", min_items=2, max_items=4
    )
    short_term_actions: List[ActionItem] = Field(
        description="Actions to take in 3-12 months", min_items=2, max_items=4
    )
    long_term_actions: List[ActionItem] = Field(
        description="Actions to take in 1-3 years", min_items=1, max_items=3
    )
    key_success_metrics: List[str] = Field(
        description="Key metrics to track success", min_items=3, max_items=5
    )
    competitive_advantages_to_build: List[str] = Field(
        description="Specific competitive advantages to develop",
        min_items=2,
        max_items=4,
    )


class CompetitiveAnalysisResult(BaseModel):
    """Complete competitive analysis result"""

    your_company: CompanyAnalysis = Field(description="Analysis of your company")
    competitor: CompanyAnalysis = Field(description="Analysis of competitor company")
    competitive_insights: CompetitiveInsights = Field(
        description="Competitive insights and recommendations"
    )


class ComprehensiveReport(BaseModel):
    """Comprehensive competitive analysis report"""

    executive_summary: str = Field(
        description="Executive summary of the competitive landscape"
    )
    company_profiles: str = Field(description="Detailed company profiles section")
    swot_analysis: str = Field(description="SWOT analysis section")
    technology_comparison: str = Field(description="Technology stack comparison")
    competitive_positioning: str = Field(description="Competitive positioning analysis")
    strategic_recommendations: str = Field(
        description="Strategic recommendations section"
    )
    market_opportunities: str = Field(description="Market opportunities section")


# Initialize OpenAI LLM
def get_llm():
    """Initialize and cache the OpenAI LLM"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error(
            "OpenAI API key not found. Please set it in Streamlit secrets or environment variables."
        )
        st.stop()

    return ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o",
        temperature=0.0,
    )


# Initialize Tavily search tool
def get_search_tool():
    """Initialize and cache the search tool"""
    try:
        return TavilySearchResults(
            max_results=10,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
        )
    except Exception as e:
        st.error(f"❌ Error initializing search tool: {str(e)}")
        return None


def create_positioning_map(
    analysis_data: CompetitiveAnalysisResult, x_metric="price", y_metric="innovation"
):
    """Create a 2x2 competitive positioning map"""
    try:
        # Extract positioning scores
        your_company_data = analysis_data.your_company
        competitor_data = analysis_data.competitor

        # Create dataframe for plotting
        df = pd.DataFrame(
            {
                "company": [your_company_data.name, competitor_data.name],
                "x_score": [
                    getattr(your_company_data.positioning_scores, x_metric),
                    getattr(competitor_data.positioning_scores, x_metric),
                ],
                "y_score": [
                    getattr(your_company_data.positioning_scores, y_metric),
                    getattr(competitor_data.positioning_scores, y_metric),
                ],
                "color": ["#1f77b4", "#ff7f0e"],  # Different colors for each company
            }
        )

        # Create scatter plot
        fig = go.Figure()

        for i, row in df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["x_score"]],
                    y=[row["y_score"]],
                    mode="markers+text",
                    marker=dict(size=20, color=row["color"]),
                    text=row["company"],
                    textposition="top center",
                    name=row["company"],
                    showlegend=True,
                )
            )

        # Add quadrant lines
        fig.add_hline(y=5.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=5.5, line_dash="dash", line_color="gray", opacity=0.5)

        # Update layout
        fig.update_layout(
            title=f"Competitive Positioning: {x_metric.title()} vs {y_metric.title()}",
            xaxis_title=f"{x_metric.title()} (1-10)",
            yaxis_title=f"{y_metric.title()} (1-10)",
            xaxis=dict(range=[0, 11], showgrid=True),
            yaxis=dict(range=[0, 11], showgrid=True),
            height=500,
            width=600,
            template="plotly_white",
        )

        # Add quadrant labels
        fig.add_annotation(
            x=2.5,
            y=8.5,
            text="High Innovation<br>Low Price",
            showarrow=False,
            opacity=0.7,
        )
        fig.add_annotation(
            x=8.5,
            y=8.5,
            text="High Innovation<br>High Price",
            showarrow=False,
            opacity=0.7,
        )
        fig.add_annotation(
            x=2.5,
            y=2.5,
            text="Low Innovation<br>Low Price",
            showarrow=False,
            opacity=0.7,
        )
        fig.add_annotation(
            x=8.5,
            y=2.5,
            text="Low Innovation<br>High Price",
            showarrow=False,
            opacity=0.7,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating positioning map: {str(e)}")
        return None


def create_radar_chart(analysis_data: CompetitiveAnalysisResult):
    """Create radar chart comparing both companies across multiple metrics"""
    try:
        your_company_data = analysis_data.your_company
        competitor_data = analysis_data.competitor

        categories = ["price", "innovation", "market_share", "customer_satisfaction"]
        your_scores = [
            getattr(your_company_data.positioning_scores, cat) for cat in categories
        ]
        competitor_scores = [
            getattr(competitor_data.positioning_scores, cat) for cat in categories
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=your_scores,
                theta=categories,
                fill="toself",
                name=your_company_data.name,
                line_color="#1f77b4",
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=competitor_scores,
                theta=categories,
                fill="toself",
                name=competitor_data.name,
                line_color="#ff7f0e",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="Competitive Radar Chart",
            height=500,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        return None


def display_action_items(action_items: List[ActionItem], timeline_label: str):
    """Display action items in a structured format"""
    st.markdown(f"### {timeline_label}")

    for i, action in enumerate(action_items, 1):
        with st.container():
            # Priority badge styling
            priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}. {action.title}**")
            with col2:
                st.markdown(
                    f"{priority_colors.get(action.priority, '⚪')} {action.priority} Priority"
                )

            st.markdown(f"**Description:** {action.description}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Timeline:** {action.timeline}")
                st.markdown(f"**Resources:** {', '.join(action.resources_needed)}")
            with col2:
                st.markdown(f"**Expected Impact:** {action.expected_impact}")

            st.markdown("---")


def extract_text_from_pdf_pypdf2(file):
    """Extract text from PDF using PyPDF2 library."""
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return (
            text.strip() if text.strip() else "No text could be extracted from the PDF."
        )
    except Exception as e:
        error_msg = f"Error extracting text from PDF using PyPDF2: {str(e)}"
        st.error(error_msg)
        return error_msg


def pdf_to_text(uploaded_file):
    """Main function to convert PDF to text."""
    try:
        raw_text = extract_text_from_pdf_pypdf2(uploaded_file)
        if "Error" in raw_text or len(raw_text.strip()) < 50:
            return f"Could not extract meaningful text from PDF. Raw extraction result: {raw_text}"
        return raw_text
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        st.error(error_msg)
        return error_msg


def safe_display_search_results(search_results):
    """Safely display search results, handling various data types and formats."""
    try:
        if isinstance(search_results, str):
            st.text(search_results)
            return
        if isinstance(search_results, (dict, list)):
            st.json(search_results)
            return
        if hasattr(search_results, "__dict__"):
            result_dict = vars(search_results)
            st.json(result_dict)
            return
        st.text(str(search_results))
    except Exception as e:
        st.error(f"Error displaying search results: {str(e)}")
        st.text("Raw search results:")
        st.text(str(search_results))
