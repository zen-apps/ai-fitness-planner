import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict
import os
import PyPDF2
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Competitive Analysis",
    initial_sidebar_state="collapsed",
)

from dotenv import load_dotenv

load_dotenv("./config/dev.env")


from helpers.competitive_helper import (
    PositioningScores,
    CompanyAnalysis,
    CompetitiveInsights,
    ActionItem,
    ActionPlan,
    CompetitiveAnalysisResult,
    ComprehensiveReport,
    get_llm,
    get_search_tool,
    create_positioning_map,
    create_radar_chart,
    display_action_items,
    extract_text_from_pdf_pypdf2,
    pdf_to_text,
    safe_display_search_results,
)


def initialize_session_state():
    """Initialize session state variables."""
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "action_plan" not in st.session_state:
        st.session_state.action_plan = None
    if "report_data" not in st.session_state:
        st.session_state.report_data = None
    if "your_company_results" not in st.session_state:
        st.session_state.your_company_results = None
    if "competitor_results" not in st.session_state:
        st.session_state.competitor_results = None
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = "No PDF was provided."
    if "your_company" not in st.session_state:
        st.session_state.your_company = ""
    if "competitor_company" not in st.session_state:
        st.session_state.competitor_company = ""


def generate_competitive_analysis(
    your_company: str,
    competitor_company: str,
    your_company_results: str,
    competitor_results: str,
) -> CompetitiveAnalysisResult:
    """Generate competitive analysis using structured output"""
    llm = get_llm()

    # Truncate search results to avoid token limits
    max_result_length = 2000
    your_results_truncated = (
        str(your_company_results)[:max_result_length] + "..."
        if len(str(your_company_results)) > max_result_length
        else str(your_company_results)
    )
    competitor_results_truncated = (
        str(competitor_results)[:max_result_length] + "..."
        if len(str(competitor_results)) > max_result_length
        else str(competitor_results)
    )

    prompt = f"""
    You are a business analyst conducting competitive analysis.

    Search results for {your_company}:
    {your_results_truncated}

    Search results for {competitor_company}:
    {competitor_results_truncated}

    Provide a concise competitive analysis for both companies:
    
    Guidelines:
    - Keep each SWOT item to 1-2 sentences max
    - Use brief, factual descriptions
    - Score positioning metrics (1-10): price competitiveness, innovation level, market position, customer satisfaction
    - Identify 2-3 key differentiators and market gaps
    - Provide 3-4 strategic recommendations
    
    Be concise and focus on key insights from the search results.
    """

    structured_llm = llm.with_structured_output(CompetitiveAnalysisResult)
    response = structured_llm.invoke([HumanMessage(content=prompt)])

    return response


def generate_action_plan(
    analysis_data: CompetitiveAnalysisResult,
    your_company: str,
    competitor_company: str,
) -> ActionPlan:
    """Generate detailed action plan to beat the competition"""
    llm = get_llm()

    # Create a comprehensive summary for action plan generation
    analysis_summary = f"""
    YOUR COMPANY: {your_company}
    Overview: {analysis_data.your_company.overview}
    
    Strengths: {', '.join(analysis_data.your_company.strengths)}
    Weaknesses: {', '.join(analysis_data.your_company.weaknesses)}
    Opportunities: {', '.join(analysis_data.your_company.opportunities)}
    Threats: {', '.join(analysis_data.your_company.threats)}
    
    Tech Stack: {', '.join(analysis_data.your_company.tech_stack)}
    Positioning Scores - Price: {analysis_data.your_company.positioning_scores.price}/10, 
    Innovation: {analysis_data.your_company.positioning_scores.innovation}/10,
    Market Share: {analysis_data.your_company.positioning_scores.market_share}/10,
    Customer Satisfaction: {analysis_data.your_company.positioning_scores.customer_satisfaction}/10
    
    COMPETITOR: {competitor_company}
    Overview: {analysis_data.competitor.overview}
    
    Strengths: {', '.join(analysis_data.competitor.strengths)}
    Weaknesses: {', '.join(analysis_data.competitor.weaknesses)}
    
    Tech Stack: {', '.join(analysis_data.competitor.tech_stack)}
    Positioning Scores - Price: {analysis_data.competitor.positioning_scores.price}/10,
    Innovation: {analysis_data.competitor.positioning_scores.innovation}/10,
    Market Share: {analysis_data.competitor.positioning_scores.market_share}/10,
    Customer Satisfaction: {analysis_data.competitor.positioning_scores.customer_satisfaction}/10
    
    KEY DIFFERENTIATORS: {', '.join(analysis_data.competitive_insights.key_differentiators)}
    MARKET GAPS: {', '.join(analysis_data.competitive_insights.market_gaps)}
    """

    prompt = f"""
    You are a strategic business consultant creating an actionable plan for {your_company} to outcompete {competitor_company}.

    Based on this competitive analysis:
    {analysis_summary}

    Create a comprehensive action plan with specific, measurable actions that {your_company} can take to gain competitive advantage.

    Guidelines for action items:
    - Be specific and actionable (avoid vague recommendations)
    - Include realistic timelines
    - Consider resource requirements
    - Focus on areas where {your_company} can realistically outperform {competitor_company}
    - Target competitor weaknesses and market gaps
    - Build on {your_company}'s existing strengths
    - Address {your_company}'s key weaknesses that are holding it back

    Prioritize actions based on:
    - Immediate wins (30-90 days): Quick improvements with high impact
    - Short-term strategic moves (3-12 months): Building capabilities and market position
    - Long-term competitive advantages (1-3 years): Fundamental differentiation

    Make recommendations practical and executable for a real business.
    """

    structured_llm = llm.with_structured_output(ActionPlan)
    response = structured_llm.invoke([HumanMessage(content=prompt)])

    return response


def generate_comprehensive_report(
    analysis_data: CompetitiveAnalysisResult,
    pdf_content: str,
    your_company: str,
    competitor_company: str,
) -> ComprehensiveReport:
    """Generate comprehensive report using structured output"""
    llm = get_llm()

    # Truncate PDF content to avoid token limits
    max_pdf_length = 1000
    pdf_truncated = (
        pdf_content[:max_pdf_length] + "..."
        if len(pdf_content) > max_pdf_length
        else pdf_content
    )

    # Use a simpler data representation
    analysis_summary = f"""
    {your_company}: {analysis_data.your_company.overview}
    Strengths: {', '.join(analysis_data.your_company.strengths[:2])}
    Positioning: Innovation {analysis_data.your_company.positioning_scores.innovation}, Price {analysis_data.your_company.positioning_scores.price}
    
    {competitor_company}: {analysis_data.competitor.overview}  
    Strengths: {', '.join(analysis_data.competitor.strengths[:2])}
    Positioning: Innovation {analysis_data.competitor.positioning_scores.innovation}, Price {analysis_data.competitor.positioning_scores.price}
    
    Key Insights: {', '.join(analysis_data.competitive_insights.key_differentiators)}
    """

    prompt = f"""
    Create a concise competitive analysis report based on:
    
    Analysis Summary: {analysis_summary}
    
    PDF Context: {pdf_truncated}

    Generate brief sections (2-3 sentences each) for:
    1. Executive Summary
    2. Company Profiles for {your_company} and {competitor_company}
    3. SWOT Analysis comparison
    4. Technology Comparison
    5. Competitive Positioning
    6. Strategic Recommendations (3-4 actionable items for {your_company})
    7. Market Opportunities (2-3 key gaps)

    Keep each section concise and focused on key insights.
    """

    structured_llm = llm.with_structured_output(ComprehensiveReport)
    response = structured_llm.invoke([HumanMessage(content=prompt)])

    return response


def run_analysis(your_company: str, competitor_company: str, uploaded_file):
    """Run the complete analysis and store results in session state."""
    # Initialize tools
    search = get_search_tool()

    if not search:
        st.error(
            "❌ Could not initialize search tool. Please check your Tavily API key."
        )
        return

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Search for your company
        status_text.text(f"🔍 Researching {your_company}...")
        progress_bar.progress(15)
        your_company_results = search.invoke(
            f"{your_company} recent news company analysis"
        )
        st.session_state.your_company_results = your_company_results

        # Step 2: Search for competitor
        status_text.text(f"🔍 Researching {competitor_company}...")
        progress_bar.progress(30)
        competitor_results = search.invoke(
            f"{competitor_company} recent news company analysis"
        )
        st.session_state.competitor_results = competitor_results

        # Step 3: Generate competitive analysis using structured output
        status_text.text("📊 Generating competitive analysis...")
        progress_bar.progress(50)

        analysis_data = generate_competitive_analysis(
            your_company,
            competitor_company,
            str(your_company_results),
            str(competitor_results),
        )
        st.session_state.analysis_data = analysis_data

        # st.success("✅ Analysis data generated successfully!")

        # Step 4: Generate action plan
        status_text.text("🎯 Creating strategic action plan...")
        progress_bar.progress(65)

        action_plan = generate_action_plan(
            analysis_data, your_company, competitor_company
        )
        st.session_state.action_plan = action_plan

        # Step 5: Process PDF if uploaded
        pdf_text = "No PDF was provided."
        if uploaded_file:
            status_text.text("📄 Processing PDF document...")
            progress_bar.progress(75)
            pdf_text = pdf_to_text(uploaded_file)
        st.session_state.pdf_text = pdf_text

        # Step 6: Generate comprehensive report
        status_text.text("📋 Generating comprehensive report...")
        progress_bar.progress(90)

        report_data = generate_comprehensive_report(
            analysis_data, pdf_text, your_company, competitor_company
        )
        st.session_state.report_data = report_data

        # Complete
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)

        # Store company names in session state
        st.session_state.your_company = your_company
        st.session_state.competitor_company = competitor_company

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your API key and try again.")

    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()


def display_results():
    """Display the analysis results using data from session state."""
    if not st.session_state.analysis_data:
        return

    analysis_data = st.session_state.analysis_data
    action_plan = st.session_state.action_plan
    report_data = st.session_state.report_data
    your_company = st.session_state.your_company
    competitor_company = st.session_state.competitor_company

    # Display results
    st.header("📊 Competitive Analysis Results")

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📈 Overview",
            "🎯 SWOT Analysis",
            "💭 Sentiment Analysis",
            "🔧 Tech Comparison",
            "📍 Positioning Maps",
            "🚀 Action Plan",
        ]
    )

    with tab1:
        st.subheader("Executive Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {your_company}")
            st.write(analysis_data.your_company.overview)

        with col2:
            st.markdown(f"### {competitor_company}")
            st.write(analysis_data.competitor.overview)

        st.subheader("Key Competitive Insights")
        insights = analysis_data.competitive_insights

        st.markdown("**Key Differentiators:**")
        for diff in insights.key_differentiators:
            st.write(f"• {diff}")

        st.markdown("**Strategic Recommendations:**")
        for rec in insights.strategic_recommendations:
            st.write(f"• {rec}")

    with tab2:
        st.subheader("SWOT Analysis Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {your_company}")

            st.markdown("**Strengths** 💪")
            for strength in analysis_data.your_company.strengths:
                st.write(f"• {strength}")

            st.markdown("**Weaknesses** ⚠️")
            for weakness in analysis_data.your_company.weaknesses:
                st.write(f"• {weakness}")

            st.markdown("**Opportunities** 🌟")
            for opportunity in analysis_data.your_company.opportunities:
                st.write(f"• {opportunity}")

            st.markdown("**Threats** ⚡")
            for threat in analysis_data.your_company.threats:
                st.write(f"• {threat}")

        with col2:
            st.markdown(f"### {competitor_company}")

            st.markdown("**Strengths** 💪")
            for strength in analysis_data.competitor.strengths:
                st.write(f"• {strength}")

            st.markdown("**Weaknesses** ⚠️")
            for weakness in analysis_data.competitor.weaknesses:
                st.write(f"• {weakness}")

            st.markdown("**Opportunities** 🌟")
            for opportunity in analysis_data.competitor.opportunities:
                st.write(f"• {opportunity}")

            st.markdown("**Threats** ⚡")
            for threat in analysis_data.competitor.threats:
                st.write(f"• {threat}")

    with tab3:
        st.subheader("Recent News Sentiment Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {your_company}")
            sentiment = analysis_data.your_company.recent_news_sentiment

            if sentiment.lower() == "positive":
                st.success(f"Sentiment: {sentiment.upper()} 😊")
            elif sentiment.lower() == "negative":
                st.error(f"Sentiment: {sentiment.upper()} 😟")
            else:
                st.info(f"Sentiment: {sentiment.upper()} 😐")

            st.write(analysis_data.your_company.sentiment_explanation)

        with col2:
            st.markdown(f"### {competitor_company}")
            sentiment = analysis_data.competitor.recent_news_sentiment

            if sentiment.lower() == "positive":
                st.success(f"Sentiment: {sentiment.upper()} 😊")
            elif sentiment.lower() == "negative":
                st.error(f"Sentiment: {sentiment.upper()} 😟")
            else:
                st.info(f"Sentiment: {sentiment.upper()} 😐")

            st.write(analysis_data.competitor.sentiment_explanation)

    with tab4:
        st.subheader("Technology Stack Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {your_company} Tech Stack")
            for tech in analysis_data.your_company.tech_stack:
                st.write(f"• {tech}")

        with col2:
            st.markdown(f"### {competitor_company} Tech Stack")
            for tech in analysis_data.competitor.tech_stack:
                st.write(f"• {tech}")

    with tab5:
        st.subheader("Competitive Positioning Maps")

        # Positioning map options - Now outside of form since data is in session state
        col1, col2 = st.columns(2)
        with col1:
            x_metric = st.selectbox(
                "X-Axis Metric",
                [
                    "price",
                    "innovation",
                    "market_share",
                    "customer_satisfaction",
                ],
                key="x_metric_select",
            )
        with col2:
            y_metric = st.selectbox(
                "Y-Axis Metric",
                [
                    "innovation",
                    "price",
                    "market_share",
                    "customer_satisfaction",
                ],
                key="y_metric_select",
            )

        # Create and display positioning map
        if x_metric != y_metric:
            positioning_fig = create_positioning_map(analysis_data, x_metric, y_metric)
            if positioning_fig:
                st.plotly_chart(positioning_fig, use_container_width=True)
        else:
            st.warning("Please select different metrics for X and Y axes.")

        st.subheader("Radar Chart Comparison")
        radar_fig = create_radar_chart(analysis_data)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)

    with tab6:
        st.subheader("🚀 Strategic Action Plan")
        st.markdown(
            f"**Comprehensive plan for {your_company} to outcompete {competitor_company}**"
        )

        # Display immediate actions
        display_action_items(
            action_plan.immediate_actions, "🔥 Immediate Actions (30-90 Days)"
        )

        # Display short-term actions
        display_action_items(
            action_plan.short_term_actions,
            "📈 Short-Term Actions (3-12 Months)",
        )

        # Display long-term actions
        display_action_items(
            action_plan.long_term_actions, "🎯 Long-Term Actions (1-3 Years)"
        )

        # Key Success Metrics
        st.subheader("📊 Key Success Metrics to Track")
        for i, metric in enumerate(action_plan.key_success_metrics, 1):
            st.write(f"{i}. {metric}")

        # Competitive Advantages to Build
        st.subheader("🏆 Competitive Advantages to Develop")
        for i, advantage in enumerate(action_plan.competitive_advantages_to_build, 1):
            st.write(f"{i}. {advantage}")

        # Action Plan Summary
        st.subheader("📋 Action Plan Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Immediate Actions", len(action_plan.immediate_actions))
            st.metric(
                "High Priority Items",
                len(
                    [
                        action
                        for action in action_plan.immediate_actions
                        + action_plan.short_term_actions
                        if action.priority == "High"
                    ]
                ),
            )

        with col2:
            st.metric("Short-Term Actions", len(action_plan.short_term_actions))
            st.metric(
                "Medium Priority Items",
                len(
                    [
                        action
                        for action in action_plan.immediate_actions
                        + action_plan.short_term_actions
                        + action_plan.long_term_actions
                        if action.priority == "Medium"
                    ]
                ),
            )

        with col3:
            st.metric("Long-Term Actions", len(action_plan.long_term_actions))
            st.metric("Success Metrics", len(action_plan.key_success_metrics))

    # Full Report Section
    st.header("📋 Comprehensive Report")

    # Display each section of the report
    st.subheader("Executive Summary")
    st.markdown(report_data.executive_summary)

    st.subheader("Company Profiles")
    st.markdown(report_data.company_profiles)

    st.subheader("SWOT Analysis")
    st.markdown(report_data.swot_analysis)

    st.subheader("Technology Comparison")
    st.markdown(report_data.technology_comparison)

    st.subheader("Competitive Positioning")
    st.markdown(report_data.competitive_positioning)

    st.subheader("Strategic Recommendations")
    st.markdown(report_data.strategic_recommendations)

    st.subheader("Market Opportunities")
    st.markdown(report_data.market_opportunities)

    # Raw data expandable sections
    with st.expander("🔍 View Raw Search Results", expanded=False):
        st.subheader(f"Search Results for {your_company}")
        safe_display_search_results(st.session_state.your_company_results)

        st.subheader(f"Search Results for {competitor_company}")
        safe_display_search_results(st.session_state.competitor_results)

    if st.session_state.pdf_text and "Error" not in st.session_state.pdf_text:
        with st.expander("📄 PDF Content Preview", expanded=False):
            st.text_area(
                "Extracted Text",
                (
                    st.session_state.pdf_text[:1000] + "..."
                    if len(st.session_state.pdf_text) > 1000
                    else st.session_state.pdf_text
                ),
                height=200,
            )
            st.info(f"Extracted {len(st.session_state.pdf_text)} characters from PDF")

    # Download options
    st.subheader("📥 Download Results")
    col1, col2, col3 = st.columns(3)

    # Combine all report sections for full report download
    full_report = f"""# Competitive Analysis Report

## Executive Summary
{report_data.executive_summary}

## Company Profiles
{report_data.company_profiles}

## SWOT Analysis
{report_data.swot_analysis}

## Technology Comparison
{report_data.technology_comparison}

## Competitive Positioning
{report_data.competitive_positioning}

## Strategic Recommendations
{report_data.strategic_recommendations}

## Market Opportunities
{report_data.market_opportunities}

## Action Plan

### Immediate Actions (30-90 Days)
{chr(10).join([f"**{action.title}** ({action.priority} Priority, {action.timeline}): {action.description}" for action in action_plan.immediate_actions])}

### Short-Term Actions (3-12 Months)
{chr(10).join([f"**{action.title}** ({action.priority} Priority, {action.timeline}): {action.description}" for action in action_plan.short_term_actions])}

### Long-Term Actions (1-3 Years)
{chr(10).join([f"**{action.title}** ({action.priority} Priority, {action.timeline}): {action.description}" for action in action_plan.long_term_actions])}

### Key Success Metrics
{chr(10).join([f"- {metric}" for metric in action_plan.key_success_metrics])}

### Competitive Advantages to Build
{chr(10).join([f"- {advantage}" for advantage in action_plan.competitive_advantages_to_build])}
"""

    # Create action plan text for download
    immediate_actions_text = "\n".join(
        [
            f"{i+1}. **{action.title}** ({action.priority} Priority)\n"
            f"   Timeline: {action.timeline}\n"
            f"   Description: {action.description}\n"
            f"   Resources: {', '.join(action.resources_needed)}\n"
            f"   Expected Impact: {action.expected_impact}\n"
            for i, action in enumerate(action_plan.immediate_actions)
        ]
    )

    short_term_actions_text = "\n".join(
        [
            f"{i+1}. **{action.title}** ({action.priority} Priority)\n"
            f"   Timeline: {action.timeline}\n"
            f"   Description: {action.description}\n"
            f"   Resources: {', '.join(action.resources_needed)}\n"
            f"   Expected Impact: {action.expected_impact}\n"
            for i, action in enumerate(action_plan.short_term_actions)
        ]
    )

    long_term_actions_text = "\n".join(
        [
            f"{i+1}. **{action.title}** ({action.priority} Priority)\n"
            f"   Timeline: {action.timeline}\n"
            f"   Description: {action.description}\n"
            f"   Resources: {', '.join(action.resources_needed)}\n"
            f"   Expected Impact: {action.expected_impact}\n"
            for i, action in enumerate(action_plan.long_term_actions)
        ]
    )

    action_plan_text = f"""# Action Plan for {your_company}

## Immediate Actions (30-90 Days)
{immediate_actions_text}

## Short-Term Actions (3-12 Months)
{short_term_actions_text}

## Long-Term Actions (1-3 Years)
{long_term_actions_text}

## Key Success Metrics
{chr(10).join([f"- {metric}" for metric in action_plan.key_success_metrics])}

## Competitive Advantages to Build
{chr(10).join([f"- {advantage}" for advantage in action_plan.competitive_advantages_to_build])}
"""

    with col1:
        st.download_button(
            label="📄 Download Full Report (Markdown)",
            data=full_report,
            file_name=f"{your_company}_vs_{competitor_company}_analysis.md",
            mime="text/markdown",
        )

    with col2:
        st.download_button(
            label="🚀 Download Action Plan (Markdown)",
            data=action_plan_text,
            file_name=f"{your_company}_action_plan.md",
            mime="text/markdown",
        )

    with col3:
        # Combine analysis data with action plan for JSON download
        complete_data = {
            "competitive_analysis": analysis_data.model_dump(),
            "action_plan": action_plan.model_dump(),
        }
        st.download_button(
            label="📊 Download Complete Data (JSON)",
            data=json.dumps(complete_data, indent=2),
            file_name=f"{your_company}_vs_{competitor_company}_complete_data.json",
            mime="application/json",
        )


def main():
    # Initialize session state
    initialize_session_state()

    st.title("🔍 Competitive Analysis Dashboard")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check if API keys are available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info(
                "Add your OpenAI API key to Streamlit secrets or environment variables"
            )

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            st.success("✅ Tavily Search Connected")
        else:
            st.error("❌ Tavily API Key Missing")
            st.info(
                "Add your Tavily API key to environment variables for web search functionality"
            )

        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown(
            """
        1. Enter your company name
        2. Enter competitor company name
        3. Optionally upload a PDF document
        4. Click 'Start Analysis' to begin
        5. View SWOT analysis, sentiment analysis, tech comparison, positioning maps, and actionable plans
        """
        )

        st.markdown("---")
        st.markdown("### 📈 Analysis Features")
        st.info(
            """
        • SWOT Analysis for both companies
        • Sentiment analysis from recent news
        • Technology stack comparison  
        • 2x2 Competitive positioning map
        • Radar chart comparison
        • Strategic recommendations
        • **NEW:** Detailed Action Plan with timelines
        """
        )

        # Clear results button
        if st.button("🗑️ Clear All Results"):
            for key in [
                "analysis_data",
                "action_plan",
                "report_data",
                "your_company_results",
                "competitor_results",
                "pdf_text",
                "your_company",
                "competitor_company",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Main input section wrapped in form
    with st.form(key="analysis_form"):
        col1, col2 = st.columns(2)

        with col1:
            your_company = st.text_input(
                "🏢 Your Company Name",
                placeholder="e.g., Your startup, Microsoft, etc.",
                value=st.session_state.get("your_company", ""),
            )

        with col2:
            competitor_company = st.text_input(
                "🎯 Competitor Company Name",
                placeholder="e.g., Apple, Google, Tesla, etc.",
                value=st.session_state.get("competitor_company", ""),
            )

        uploaded_file = st.file_uploader(
            "📄 Upload PDF (optional)",
            type=["pdf"],
            help="Upload company documents, reports, or any relevant PDF for additional context",
        )

        start_analysis_button = st.form_submit_button(
            "🚀 Start Competitive Analysis",
            type="primary",
        )

        if (
            start_analysis_button
            and your_company.strip()
            and competitor_company.strip()
        ):
            run_analysis(
                your_company.strip(), competitor_company.strip(), uploaded_file
            )

    # Display results if they exist
    if st.session_state.analysis_data:
        display_results()
    elif not st.session_state.analysis_data and not (
        your_company.strip() and competitor_company.strip()
    ):
        st.info(
            "👆 Please enter both company names above to start the competitive analysis."
        )


if __name__ == "__main__":
    main()
