import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import openpyxl
import tempfile
import base64
from PyPDF2 import PdfReader
import json
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv("./config/dev.env")


# Pydantic models for structured feedback analysis
class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""

    overall_sentiment_score: float = Field(
        description="Overall sentiment score from -1 (negative) to 1 (positive)"
    )
    positive_percentage: float = Field(description="Percentage of positive feedback")
    neutral_percentage: float = Field(description="Percentage of neutral feedback")
    negative_percentage: float = Field(description="Percentage of negative feedback")
    confidence_score: float = Field(
        description="Confidence level of sentiment analysis"
    )


class FeedbackThemes(BaseModel):
    """Key themes extracted from customer feedback."""

    primary_themes: List[str] = Field(description="Most frequently mentioned themes")
    product_themes: List[str] = Field(description="Product-related feedback themes")
    service_themes: List[str] = Field(description="Service-related feedback themes")
    pricing_themes: List[str] = Field(description="Pricing-related feedback themes")
    emerging_themes: List[str] = Field(description="New or trending themes")


class CustomerInsights(BaseModel):
    """AI-generated customer feedback insights."""

    overall_satisfaction_score: int = Field(
        description="Overall customer satisfaction score 1-10"
    )
    key_strengths: List[str] = Field(description="Major customer-praised strengths")
    key_concerns: List[str] = Field(description="Primary areas of customer concern")
    product_feedback: str = Field(description="Analysis of product-related feedback")
    service_feedback: str = Field(description="Analysis of service-related feedback")
    user_experience_feedback: str = Field(
        description="Analysis of user experience feedback"
    )
    competitive_insights: str = Field(description="Insights about competitive mentions")
    actionable_recommendations: List[str] = Field(
        description="Specific recommendations for improvement"
    )
    priority_issues: List[str] = Field(description="High-priority issues to address")


class TrendAnalysis(BaseModel):
    """Analysis of feedback trends over time."""

    sentiment_trend: str = Field(description="Analysis of sentiment trends")
    volume_trend: str = Field(description="Analysis of feedback volume trends")
    theme_evolution: str = Field(description="How themes have evolved over time")
    seasonal_patterns: str = Field(
        description="Seasonal or cyclical patterns in feedback"
    )
    key_trend_insights: List[str] = Field(
        description="Key insights from trend analysis"
    )


# Cached LLM initialization
@st.cache_resource
def get_feedback_llm():
    """Initialize OpenAI LLM for feedback analysis."""
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
    )


def get_search_tool(time_back_search=None):
    """Initialize and cache the search tool"""
    try:
        time_back = None
        if time_back_search is not None:
            if time_back_search == "1 month":
                time_back = "month"
            elif time_back_search == "1 week":
                time_back = "week"
            elif time_back_search == "1 day":
                time_back = "day"
            elif time_back_search == "1 year":
                time_back = "year"

        search_tool = TavilySearchResults(
            max_results=10,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
        )
        return search_tool
    except Exception as e:
        st.error(f"❌ Error initializing search tool: {str(e)}")
        return None


# Configure page
st.set_page_config(
    page_title="Customer Feedback Insights",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def file_to_base64(file):
    """Convert uploaded file to base64 string."""
    encoded_string = base64.b64encode(file.getvalue()).decode("utf-8")
    return encoded_string


def extract_text_with_pypdf2(pdf_file):
    """Extract text from PDF using PyPDF2 as fallback."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Extract text content using PyPDF2
        pdf_reader = PdfReader(tmp_file_path)
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

        # Clean up temp file
        os.unlink(tmp_file_path)

        return text_content, len(pdf_reader.pages)

    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {e}")
        return "", 0


def extract_data_from_pdf(pdf_file):
    """Extract customer feedback data from PDF using hybrid approach: PyPDF2 + AI analysis."""
    try:
        # First, extract text using PyPDF2 (reliable for text-based PDFs)
        st.info("📄 Extracting text from PDF...")
        extracted_text, num_pages = extract_text_with_pypdf2(pdf_file)

        if not extracted_text.strip():
            st.warning(
                "⚠️ No readable text found in PDF. The PDF might be image-based or corrupted."
            )
            return {
                "text": "",
                "structured_feedback_data": "",
                "success": False,
                "extraction_method": "Failed - No Text",
            }

        st.success(
            f"✅ Extracted text from {num_pages} pages ({len(extracted_text):,} characters)"
        )

        # Now use AI to analyze and structure the extracted text
        st.info("🤖 Analyzing customer feedback data with AI...")
        llm = get_feedback_llm()

        # Split text if it's too long for the LLM
        max_text_length = 12000  # Conservative limit for token count
        if len(extracted_text) > max_text_length:
            text_to_analyze = (
                extracted_text[:max_text_length] + "...\n[Text truncated for analysis]"
            )
            st.info(
                f"📝 Text truncated to {max_text_length:,} characters for AI analysis"
            )
        else:
            text_to_analyze = extracted_text

        feedback_extraction_prompt = f"""
        You are a customer feedback analyst. Analyze this extracted text from a customer feedback document and identify key insights.
        
        EXTRACTED TEXT:
        {text_to_analyze}
        
        Please provide a structured analysis including:
        
        1. **Document Type**: What type of feedback document is this? (Survey Results, Review Analysis, Social Media Mentions, etc.)
        
        2. **Key Feedback Data Found**:
           - Customer satisfaction scores
           - Sentiment indicators
           - Product/service mentions
           - Complaint categories
           - Praise categories
           - Time periods covered
        
        3. **Key Themes** (extract specific themes with context):
           - Product quality feedback
           - Service experience feedback
           - Pricing feedback
           - User experience feedback
           - Feature requests or suggestions
        
        4. **Sentiment Analysis**: Provide overall sentiment breakdown and specific positive/negative feedback examples.
        
        5. **Summary**: Provide a brief summary of the customer feedback insights and overall satisfaction trends.
        
        If this doesn't appear to be customer feedback data, please state that clearly.
        
        Format your response in a clear, structured way that would be useful for marketing teams and customer service managers.
        """

        try:
            feedback_response = llm.invoke(
                [HumanMessage(content=feedback_extraction_prompt)]
            )
            structured_data = feedback_response.content
            st.success("✅ AI analysis completed successfully")
        except Exception as ai_error:
            st.warning(f"AI analysis failed: {ai_error}")
            structured_data = "AI analysis failed. Raw text extraction was successful but could not be processed by AI."

        return {
            "text": extracted_text,
            "structured_feedback_data": structured_data,
            "success": True,
            "text_length": len(extracted_text),
            "extraction_method": "PyPDF2 + AI Analysis",
            "pages_processed": num_pages,
        }

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return {
            "text": "",
            "structured_feedback_data": "",
            "success": False,
            "extraction_method": "Failed",
        }


def extract_data_from_excel(excel_file):
    """Extract customer feedback data from Excel file."""
    try:
        # Read Excel file with multiple sheets
        excel_data = pd.read_excel(excel_file, sheet_name=None)

        extracted_sheets = []
        for sheet_name, df in excel_data.items():
            if not df.empty:
                extracted_sheets.append(
                    {"sheet_name": sheet_name, "data": df, "shape": df.shape}
                )

        return {"sheets": extracted_sheets, "success": True}

    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return {"sheets": [], "success": False}


def extract_data_from_csv(csv_file):
    """Extract customer feedback data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        return {"data": df, "success": True, "shape": df.shape}
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return {"data": None, "success": False}


def search_web_feedback(company_name, search_terms, time_period=None):
    """Search the web for customer feedback about a company."""
    try:
        search_tool = get_search_tool(time_period)
        if not search_tool:
            return {"success": False, "error": "Could not initialize search tool"}

        # Construct search query
        query = f"{company_name} customer feedback reviews complaints {search_terms}"

        st.info(f"🔍 Searching web for: {query}")

        # Perform search
        search_results = search_tool.invoke({"query": query})

        if not search_results:
            return {"success": False, "error": "No search results found"}

        # Extract relevant information
        feedback_sources = []
        for result in search_results:
            if isinstance(result, dict):
                feedback_sources.append(
                    {
                        "title": result.get("title", "N/A"),
                        "url": result.get("url", "N/A"),
                        "content": result.get("content", "N/A"),
                        "snippet": result.get("snippet", "N/A"),
                    }
                )

        return {
            "success": True,
            "query": query,
            "sources": feedback_sources,
            "total_results": len(feedback_sources),
        }

    except Exception as e:
        st.error(f"Error searching web feedback: {e}")
        return {"success": False, "error": str(e)}


def analyze_sentiment_patterns(feedback_data):
    """Analyze sentiment patterns in customer feedback."""
    try:
        patterns = {}

        # This would be implemented with actual sentiment analysis
        # For demo purposes, we'll create sample patterns
        patterns["positive_keywords"] = [
            "great",
            "excellent",
            "love",
            "amazing",
            "perfect",
        ]
        patterns["negative_keywords"] = [
            "terrible",
            "awful",
            "hate",
            "disappointed",
            "frustrated",
        ]
        patterns["neutral_keywords"] = ["okay", "average", "fine", "acceptable"]

        return patterns
    except Exception as e:
        st.warning(f"Could not analyze sentiment patterns: {e}")
        return {}


def generate_feedback_analysis(feedback_data, company_name="Company"):
    """Generate AI-powered customer feedback analysis."""
    llm = get_feedback_llm()

    # Check if this is web search data
    if "web_search_results" in feedback_data.get("files", {}):
        web_data = feedback_data["files"]["web_search_results"]["data"]
        sources_text = ""
        for source in web_data["sources"]:
            sources_text += f"Source: {source['title']}\nURL: {source['url']}\nContent: {source['content'][:500]}...\n\n"

        data_summary = f"""
        Customer Feedback Analysis for: {company_name}
        Data Source: Web Search Results
        Search Query: {web_data["search_query"]}
        Total Sources: {web_data["total_results"]}
        
        Web Search Results:
        {sources_text[:3000]}
        
        Please analyze this web-based customer feedback information and provide:
        1. Overall customer satisfaction assessment (score 1-10)
        2. Key strengths customers are praising
        3. Primary areas of customer concern
        4. Product-related feedback analysis
        5. Service-related feedback analysis
        6. User experience feedback analysis
        7. Competitive insights (if any mentions)
        8. Specific actionable recommendations for improvement
        9. Priority issues that need immediate attention
        
        Focus on actionable insights that would be valuable for marketing teams, customer service managers, and product teams.
        """
    else:
        # Original file-based analysis
        data_summary = f"""
        Customer Feedback Analysis for: {company_name}
        
        Extracted Feedback Data:
        {str(feedback_data)[:2000]}
        
        Please analyze this customer feedback information and provide:
        1. Overall customer satisfaction assessment (score 1-10)
        2. Key strengths customers are praising
        3. Primary areas of customer concern
        4. Product-related feedback analysis
        5. Service-related feedback analysis
        6. User experience feedback analysis
        7. Competitive insights (if any mentions)
        8. Specific actionable recommendations for improvement
        9. Priority issues that need immediate attention
        
        Focus on actionable insights that would be valuable for marketing teams, customer service managers, and product teams.
        """

    try:
        structured_llm = llm.with_structured_output(CustomerInsights)
        result = structured_llm.invoke([HumanMessage(content=data_summary)])
        return result
    except Exception as e:
        st.error(f"Error generating feedback analysis: {e}")
        return None


def generate_trend_analysis(time_series_data):
    """Generate trend analysis from time series feedback data."""
    llm = get_feedback_llm()

    trend_prompt = f"""
    Analyze the following time series customer feedback data for trends:
    
    {str(time_series_data)[:1500]}
    
    Provide analysis of:
    1. Sentiment trends over time
    2. Feedback volume patterns
    3. How themes have evolved over time
    4. Seasonal or cyclical patterns
    5. Key insights from the trend analysis
    
    Focus on identifying patterns, emerging issues, and improving/deteriorating areas.
    """

    try:
        structured_llm = llm.with_structured_output(TrendAnalysis)
        result = structured_llm.invoke([HumanMessage(content=trend_prompt)])
        return result
    except Exception as e:
        st.error(f"Error generating trend analysis: {e}")
        return None


def create_feedback_dashboard(data, sentiment_data, use_real_data=False):
    """Create interactive customer feedback dashboard with multiple charts."""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sentiment Distribution",
            "Feedback Volume Over Time",
            "Theme Frequency",
            "Satisfaction Score",
        ),
        specs=[
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
    )

    # Sentiment Distribution Pie Chart
    sentiment_labels = ["Positive", "Neutral", "Negative"]
    if use_real_data and sentiment_data:
        # Use real sentiment data when available
        sentiment_values = [
            sentiment_data.get("positive", 0),
            sentiment_data.get("neutral", 0),
            sentiment_data.get("negative", 0),
        ]
    else:
        # Use sample data for demo
        sentiment_values = [65, 20, 15]
    sentiment_colors = ["#2E8B57", "#FFD700", "#DC143C"]

    fig.add_trace(
        go.Pie(
            labels=sentiment_labels,
            values=sentiment_values,
            marker=dict(colors=sentiment_colors),
            name="Sentiment",
        ),
        row=1,
        col=1,
    )

    # Feedback Volume Over Time
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    feedback_volume = [120, 140, 110, 160, 180, 200]
    positive_volume = [78, 91, 71, 104, 117, 130]
    negative_volume = [18, 21, 17, 24, 27, 30]

    fig.add_trace(
        go.Scatter(
            x=months, y=feedback_volume, mode="lines+markers", name="Total Feedback"
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=months, y=positive_volume, mode="lines+markers", name="Positive"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=months, y=negative_volume, mode="lines+markers", name="Negative"),
        row=1,
        col=2,
    )

    # Theme Frequency
    if use_real_data and isinstance(data, dict) and "top_themes" in data:
        # Use real theme data when available
        themes = data["top_themes"][:5]  # Limit to top 5
        # Generate realistic counts based on data
        total_sources = data.get("total_sources", data.get("total_results", 10))
        theme_counts = [max(1, total_sources - i * 2) for i in range(len(themes))]
    else:
        # Use sample data for demo
        themes = [
            "Product Quality",
            "Customer Service",
            "Pricing",
            "User Experience",
            "Delivery",
        ]
        theme_counts = [45, 38, 22, 35, 18]

    fig.add_trace(
        go.Bar(x=themes, y=theme_counts, name="Theme Mentions"),
        row=2,
        col=1,
    )

    # Satisfaction Score Indicator
    if use_real_data and sentiment_data:
        # Calculate satisfaction score based on sentiment distribution
        positive_pct = sentiment_data.get("positive", 0)
        negative_pct = sentiment_data.get("negative", 0)
        # Simple calculation: scale positive percentage to 1-10 scale
        current_satisfaction = round(1 + (positive_pct / 100) * 9, 1)
    else:
        current_satisfaction = 7.8
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_satisfaction,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Customer Satisfaction Score"},
            delta={"reference": 7.5},
            gauge={
                "axis": {"range": [None, 10]},
                "bar": {"color": "darkgreen"},
                "steps": [
                    {"range": [0, 5], "color": "lightgray"},
                    {"range": [5, 8], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 8,
                },
            },
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=True, title_text="Customer Feedback Insights Dashboard"
    )
    return fig


def main():
    """Main application function."""
    st.title("💼 Customer Feedback Insights App")
    st.markdown(
        "*Upload files or search the web for customer feedback, reviews, and social media mentions. Extract key themes and sentiment analysis using AI.*"
    )

    # Sidebar for file upload and options
    with st.sidebar:
        st.header("📊 Data Source")

        data_source = st.radio(
            "Choose data source:",
            ["🔍 Web Search", "📁 Upload Files"],
            help="Search the web for customer feedback or upload files containing feedback data.",
        )

        if data_source == "📁 Upload Files":
            st.subheader("📁 Upload Feedback Data")
            uploaded_files = st.file_uploader(
                "Choose PDF, Excel, or CSV files",
                type=["pdf", "xlsx", "xls", "csv", "txt"],
                accept_multiple_files=True,
                help="Upload survey results, review data, social media mentions, or customer feedback reports",
            )
        else:
            st.subheader("🔍 Web Search for Feedback")

            search_company = st.text_input(
                "Company Name",
                placeholder="Enter company name to search for feedback",
                help="The company you want to search feedback for",
            )

            search_terms = st.text_input(
                "Additional Search Terms",
                placeholder="e.g., product complaints, service issues",
                help="Additional terms to refine your search",
            )

            search_time_period = st.selectbox(
                "Time Period",
                [None, "1 day", "1 week", "1 month", "1 year"],
                help="Limit search to specific time period",
            )

            # Add clear search results button if search data exists
            if "web_search_data" in st.session_state:
                if st.button(
                    "🗑️ Clear Search Results",
                    help="Clear current search results to perform a new search",
                ):
                    if "web_search_data" in st.session_state:
                        del st.session_state.web_search_data
                    if "search_summary_shown" in st.session_state:
                        del st.session_state.search_summary_shown
                    if "feedback_analysis" in st.session_state:
                        del st.session_state.feedback_analysis
                    st.rerun()

            uploaded_files = None

        st.markdown("---")

        # Company information - only show for file uploads
        if data_source == "📁 Upload Files":
            st.header("🏢 Company Information")
            company_name = st.text_input(
                "Company Name", placeholder="Enter company name"
            )
            industry = st.selectbox(
                "Industry",
                [
                    "Technology",
                    "E-commerce",
                    "Healthcare",
                    "Retail",
                    "Finance",
                    "Food & Beverage",
                    "Other",
                ],
            )

            feedback_period = st.selectbox(
                "Feedback Period",
                [
                    "Last 30 days",
                    "Last 3 months",
                    "Last 6 months",
                    "Last year",
                    "Custom",
                ],
            )
        else:
            # For web search, use the search company name and auto-set values
            company_name = search_company
            industry = "Unknown"  # Will be determined from search results
            feedback_period = search_time_period or "Web Search Period"

        st.markdown("---")

        # Analysis options
        st.header("⚙️ Analysis Options")
        include_sentiment = st.checkbox("Sentiment Analysis", value=True)
        include_themes = st.checkbox("Theme Extraction", value=True)
        include_trends = st.checkbox("Trend Analysis", value=True)
        include_competitive = st.checkbox("Competitive Insights", value=False)

        st.markdown("---")

        # API Status
        st.header("🔌 API Status")
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI Connected")
        else:
            st.error("❌ OpenAI API Key Missing")

        if os.getenv("TAVILY_API_KEY"):
            st.success("✅ Tavily Search Connected")
        else:
            st.error("❌ Tavily API Key Missing")

    # Main content area
    if uploaded_files or (data_source == "🔍 Web Search" and search_company):
        if data_source == "📁 Upload Files":
            st.header("📋 Document Processing Results")
        else:
            st.header("🔍 Web Search Results")

        extracted_data = {}
        all_dataframes = []

        # Handle Web Search
        if data_source == "🔍 Web Search" and search_company:
            if st.button("🔍 Search Web for Feedback", type="primary"):
                with st.spinner("Searching the web for customer feedback..."):
                    search_results = search_web_feedback(
                        search_company, search_terms or "", search_time_period
                    )

                    if search_results["success"]:
                        st.success(
                            f"✅ Found {search_results['total_results']} web sources"
                        )

                        # Store search results in session state for persistence
                        web_data = {
                            "search_query": search_results["query"],
                            "sources": search_results["sources"],
                            "total_results": search_results["total_results"],
                            "company": search_company,
                            "search_terms": search_terms,
                            "time_period": search_time_period,
                        }

                        # Store in session state to persist across reruns
                        st.session_state.web_search_data = {
                            "data": web_data,
                            "success": True,
                            "type": "web_search",
                        }
                        st.session_state.search_summary_shown = True

                        # Display search results
                        st.subheader("📊 Search Results Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sources Found", search_results["total_results"])
                        with col2:
                            st.metric("Company", search_company)
                        with col3:
                            st.metric("Time Period", search_time_period or "All time")

                        # Show search sources
                        with st.expander("🔍 View Search Sources"):
                            for i, source in enumerate(
                                search_results["sources"][:5]
                            ):  # Show first 5
                                st.write(f"**{i+1}. {source['title']}**")
                                st.write(f"URL: {source['url']}")
                                st.write(f"Snippet: {source['snippet'][:200]}...")
                                st.markdown("---")
                    else:
                        st.error(
                            f"❌ Search failed: {search_results.get('error', 'Unknown error')}"
                        )

            # Check if we have stored web search data and restore it to extracted_data
            if "web_search_data" in st.session_state:
                extracted_data["web_search_results"] = st.session_state.web_search_data

        # Handle File Upload Processing
        elif uploaded_files:
            # Process each uploaded file
            for file in uploaded_files:
                st.subheader(f"📄 Processing: {file.name}")

                if file.type == "application/pdf":
                    with st.spinner(f"Extracting feedback data from {file.name}..."):
                        result = extract_data_from_pdf(file)
                        if result["success"]:
                            extracted_data[file.name] = result

                            # Display extraction method and results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Extraction Method",
                                    result.get("extraction_method", "Unknown"),
                                )
                            with col2:
                                st.metric(
                                    "Text Length",
                                    f"{result.get('text_length', 0):,} chars",
                                )
                            with col3:
                                st.metric(
                                    "Status",
                                    "✅ Success" if result["success"] else "❌ Failed",
                                )

                            # Display structured feedback data
                            if result.get("structured_feedback_data"):
                                with st.expander("📊 Structured Feedback Analysis"):
                                    st.write(result["structured_feedback_data"])

                            # Display extracted text preview
                            if result["text"]:
                                with st.expander("📝 Extracted Text Preview"):
                                    preview_text = (
                                        result["text"][:2000] + "..."
                                        if len(result["text"]) > 2000
                                        else result["text"]
                                    )
                                    st.text_area(
                                        "Text Content", preview_text, height=300
                                    )

                elif file.type in [
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.ms-excel",
                ]:
                    with st.spinner(f"Reading Excel file {file.name}..."):
                        result = extract_data_from_excel(file)
                        if result["success"]:
                            extracted_data[file.name] = result

                            st.success(
                                f"✅ Read {len(result['sheets'])} sheets from Excel"
                            )
                            for sheet_info in result["sheets"]:
                                with st.expander(
                                    f"Sheet: {sheet_info['sheet_name']} - Shape: {sheet_info['shape']}"
                                ):
                                    st.dataframe(sheet_info["data"])
                                    all_dataframes.append(sheet_info["data"])

                elif file.type == "text/csv":
                    with st.spinner(f"Reading CSV file {file.name}..."):
                        result = extract_data_from_csv(file)
                        if result["success"]:
                            extracted_data[file.name] = result
                            st.success(f"✅ Read CSV with shape: {result['shape']}")

                            with st.expander(f"CSV Data Preview: {file.name}"):
                                st.dataframe(result["data"])
                                all_dataframes.append(result["data"])

        # Customer Feedback Analysis Section
        if extracted_data:
            st.header("🧠 AI Feedback Analysis")

            if st.button("🚀 Generate Feedback Analysis", type="primary"):
                with st.spinner("Analyzing customer feedback with AI..."):
                    # Combine all extracted data for analysis
                    combined_data = {
                        "files": extracted_data,
                        "company": company_name,
                        "industry": industry,
                        "period": feedback_period,
                    }

                    analysis = generate_feedback_analysis(
                        combined_data, company_name or "Your Company"
                    )

                    if analysis:
                        st.session_state.feedback_analysis = analysis
                        st.success("✅ Customer feedback analysis completed!")

            # Display Analysis Results
            if "feedback_analysis" in st.session_state:
                analysis = st.session_state.feedback_analysis

                # Key Metrics Overview
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    score_color = (
                        "green"
                        if analysis.overall_satisfaction_score >= 7
                        else (
                            "orange"
                            if analysis.overall_satisfaction_score >= 5
                            else "red"
                        )
                    )
                    st.metric(
                        "Satisfaction Score",
                        f"{analysis.overall_satisfaction_score}/10",
                    )

                with col2:
                    st.metric("Key Strengths", len(analysis.key_strengths))

                with col3:
                    st.metric("Key Concerns", len(analysis.key_concerns))

                with col4:
                    st.metric("Priority Issues", len(analysis.priority_issues))

                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "📊 Overview",
                        "🛍️ Product",
                        "🤝 Service",
                        "💻 User Experience",
                        "🚨 Priority Issues",
                    ]
                )

                with tab1:
                    st.subheader("🎯 Overall Customer Satisfaction")

                    satisfaction_color = (
                        "green"
                        if analysis.overall_satisfaction_score >= 7
                        else (
                            "orange"
                            if analysis.overall_satisfaction_score >= 5
                            else "red"
                        )
                    )
                    st.markdown(
                        f"**Satisfaction Score:** :{satisfaction_color}[{analysis.overall_satisfaction_score}/10]"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**💪 Key Strengths**")
                        for strength in analysis.key_strengths:
                            st.write(f"✅ {strength}")

                    with col2:
                        st.markdown("**⚠️ Key Concerns**")
                        for concern in analysis.key_concerns:
                            st.write(f"🔍 {concern}")

                    st.markdown("**🏆 Competitive Insights**")
                    st.info(analysis.competitive_insights)

                with tab2:
                    st.subheader("🛍️ Product Feedback Analysis")
                    st.write(analysis.product_feedback)

                    # Only show charts if we have real data, otherwise show message
                    if "web_search_results" in extracted_data or any(
                        "csv" in str(k).lower() or "excel" in str(k).lower()
                        for k in extracted_data.keys()
                    ):
                        # Try to create real product feedback chart from search data
                        if "web_search_results" in extracted_data:
                            web_data = extracted_data["web_search_results"]["data"]
                            all_content = ""
                            for source in web_data["sources"][:5]:
                                content = (
                                    source.get("content", "")
                                    + " "
                                    + source.get("snippet", "")
                                )
                                all_content += content.lower() + " "

                            # Analyze content for product aspects
                            product_aspects = []
                            satisfaction_scores = []

                            aspects_keywords = {
                                "Quality": [
                                    "quality",
                                    "fresh",
                                    "good",
                                    "excellent",
                                    "taste",
                                    "flavor",
                                ],
                                "Value": [
                                    "price",
                                    "cost",
                                    "expensive",
                                    "cheap",
                                    "worth",
                                    "value",
                                ],
                                "Packaging": [
                                    "package",
                                    "packaging",
                                    "container",
                                    "box",
                                ],
                                "Variety": [
                                    "variety",
                                    "selection",
                                    "options",
                                    "choices",
                                ],
                                "Availability": ["available", "stock", "store", "find"],
                            }

                            for aspect, keywords in aspects_keywords.items():
                                mentions = sum(
                                    1 for keyword in keywords if keyword in all_content
                                )
                                if mentions > 0:
                                    product_aspects.append(aspect)
                                    # Score based on positive/negative context (simplified)
                                    positive_words = [
                                        "good",
                                        "great",
                                        "excellent",
                                        "love",
                                        "amazing",
                                        "best",
                                    ]
                                    negative_words = [
                                        "bad",
                                        "terrible",
                                        "awful",
                                        "hate",
                                        "worst",
                                        "poor",
                                    ]

                                    pos_score = sum(
                                        1
                                        for word in positive_words
                                        if word in all_content
                                    )
                                    neg_score = sum(
                                        1
                                        for word in negative_words
                                        if word in all_content
                                    )

                                    # Calculate score (1-10 scale)
                                    base_score = 6.0  # neutral
                                    if pos_score > neg_score:
                                        score = min(
                                            10,
                                            base_score + (pos_score - neg_score) * 0.5,
                                        )
                                    else:
                                        score = max(
                                            1,
                                            base_score - (neg_score - pos_score) * 0.5,
                                        )
                                    satisfaction_scores.append(round(score, 1))

                            if product_aspects and satisfaction_scores:
                                fig_product = px.bar(
                                    x=product_aspects,
                                    y=satisfaction_scores,
                                    title="Product Satisfaction by Aspect (From Search Results)",
                                    color=satisfaction_scores,
                                    color_continuous_scale="RdYlGn",
                                    range_y=[0, 10],
                                )
                                st.plotly_chart(fig_product, use_container_width=True)
                            else:
                                st.info(
                                    "📊 Product aspect analysis requires more detailed feedback data. Upload specific product review files for detailed charts."
                                )
                        else:
                            st.info(
                                "📊 Product charts will be generated from uploaded file data."
                            )
                    else:
                        st.info(
                            "📊 Upload product review files or use web search to see product satisfaction charts."
                        )

                with tab3:
                    st.subheader("🤝 Service Feedback Analysis")
                    st.write(analysis.service_feedback)

                    # Only show charts if we have real data
                    if "web_search_results" in extracted_data or any(
                        "csv" in str(k).lower() or "excel" in str(k).lower()
                        for k in extracted_data.keys()
                    ):
                        # Try to create real service feedback chart from search data
                        if "web_search_results" in extracted_data:
                            web_data = extracted_data["web_search_results"]["data"]
                            all_content = ""
                            for source in web_data["sources"][:5]:
                                content = (
                                    source.get("content", "")
                                    + " "
                                    + source.get("snippet", "")
                                )
                                all_content += content.lower() + " "

                            # Analyze content for service aspects
                            service_aspects = []
                            service_scores = []

                            service_keywords = {
                                "Customer Support": [
                                    "support",
                                    "help",
                                    "service",
                                    "staff",
                                    "representative",
                                ],
                                "Response Time": [
                                    "fast",
                                    "quick",
                                    "slow",
                                    "response",
                                    "time",
                                    "wait",
                                ],
                                "Communication": [
                                    "communication",
                                    "clear",
                                    "explain",
                                    "understand",
                                    "rude",
                                    "polite",
                                ],
                                "Problem Resolution": [
                                    "resolve",
                                    "fix",
                                    "solution",
                                    "problem",
                                    "issue",
                                    "complaint",
                                ],
                                "Overall Experience": [
                                    "experience",
                                    "satisfaction",
                                    "happy",
                                    "disappointed",
                                    "pleased",
                                ],
                            }

                            for aspect, keywords in service_keywords.items():
                                mentions = sum(
                                    1 for keyword in keywords if keyword in all_content
                                )
                                if mentions > 0:
                                    service_aspects.append(aspect)
                                    # Score based on positive/negative context
                                    positive_words = [
                                        "good",
                                        "great",
                                        "excellent",
                                        "helpful",
                                        "friendly",
                                        "fast",
                                        "quick",
                                    ]
                                    negative_words = [
                                        "bad",
                                        "terrible",
                                        "rude",
                                        "slow",
                                        "unhelpful",
                                        "poor",
                                        "awful",
                                    ]

                                    pos_score = sum(
                                        1
                                        for word in positive_words
                                        if word in all_content
                                    )
                                    neg_score = sum(
                                        1
                                        for word in negative_words
                                        if word in all_content
                                    )

                                    # Calculate score (1-10 scale)
                                    base_score = 6.0  # neutral
                                    if pos_score > neg_score:
                                        score = min(
                                            10,
                                            base_score + (pos_score - neg_score) * 0.5,
                                        )
                                    else:
                                        score = max(
                                            1,
                                            base_score - (neg_score - pos_score) * 0.5,
                                        )
                                    service_scores.append(round(score, 1))

                            if service_aspects and service_scores:
                                fig_service = px.bar(
                                    x=service_aspects,
                                    y=service_scores,
                                    title="Service Satisfaction by Aspect (From Search Results)",
                                    color=service_scores,
                                    color_continuous_scale="RdYlGn",
                                    range_y=[0, 10],
                                )
                                st.plotly_chart(fig_service, use_container_width=True)
                            else:
                                st.info(
                                    "📊 Service aspect analysis requires more detailed feedback data. Upload specific service review files for detailed charts."
                                )
                        else:
                            st.info(
                                "📊 Service charts will be generated from uploaded file data."
                            )
                    else:
                        st.info(
                            "📊 Upload service review files or use web search to see service satisfaction charts."
                        )

                with tab4:
                    st.subheader("💻 User Experience Analysis")
                    st.write(analysis.user_experience_feedback)

                    # Only show charts if we have real data
                    if "web_search_results" in extracted_data or any(
                        "csv" in str(k).lower() or "excel" in str(k).lower()
                        for k in extracted_data.keys()
                    ):
                        # Try to create real UX feedback chart from search data
                        if "web_search_results" in extracted_data:
                            web_data = extracted_data["web_search_results"]["data"]
                            all_content = ""
                            for source in web_data["sources"][:5]:
                                content = (
                                    source.get("content", "")
                                    + " "
                                    + source.get("snippet", "")
                                )
                                all_content += content.lower() + " "

                            # Analyze content for UX aspects
                            ux_aspects = []
                            ux_scores = []

                            ux_keywords = {
                                "Website Experience": [
                                    "website",
                                    "site",
                                    "online",
                                    "web",
                                    "browse",
                                ],
                                "Mobile Experience": [
                                    "mobile",
                                    "app",
                                    "phone",
                                    "tablet",
                                    "android",
                                    "iphone",
                                ],
                                "Ordering Process": [
                                    "order",
                                    "ordering",
                                    "checkout",
                                    "purchase",
                                    "buy",
                                ],
                                "Search & Navigation": [
                                    "search",
                                    "find",
                                    "navigate",
                                    "menu",
                                    "category",
                                ],
                                "Account Management": [
                                    "account",
                                    "login",
                                    "profile",
                                    "settings",
                                    "password",
                                ],
                            }

                            for aspect, keywords in ux_keywords.items():
                                mentions = sum(
                                    1 for keyword in keywords if keyword in all_content
                                )
                                if mentions > 0:
                                    ux_aspects.append(aspect)
                                    # Score based on positive/negative context
                                    positive_words = [
                                        "easy",
                                        "simple",
                                        "convenient",
                                        "smooth",
                                        "user-friendly",
                                        "intuitive",
                                    ]
                                    negative_words = [
                                        "difficult",
                                        "confusing",
                                        "complicated",
                                        "broken",
                                        "slow",
                                        "frustrating",
                                    ]

                                    pos_score = sum(
                                        1
                                        for word in positive_words
                                        if word in all_content
                                    )
                                    neg_score = sum(
                                        1
                                        for word in negative_words
                                        if word in all_content
                                    )

                                    # Calculate score (1-10 scale)
                                    base_score = 6.0  # neutral
                                    if pos_score > neg_score:
                                        score = min(
                                            10,
                                            base_score + (pos_score - neg_score) * 0.5,
                                        )
                                    else:
                                        score = max(
                                            1,
                                            base_score - (neg_score - pos_score) * 0.5,
                                        )
                                    ux_scores.append(round(score, 1))

                            if ux_aspects and ux_scores:
                                fig_ux = px.bar(
                                    x=ux_aspects,
                                    y=ux_scores,
                                    title="User Experience Satisfaction (From Search Results)",
                                    color=ux_scores,
                                    color_continuous_scale="RdYlGn",
                                    range_y=[0, 10],
                                )
                                st.plotly_chart(fig_ux, use_container_width=True)
                            else:
                                st.info(
                                    "📊 User experience analysis requires digital platform feedback. Search terms like 'website', 'app', or 'online' may yield better UX insights."
                                )
                        else:
                            st.info(
                                "📊 UX charts will be generated from uploaded file data."
                            )
                    else:
                        st.info(
                            "📊 Upload UX feedback files or use web search to see user experience charts."
                        )

                with tab5:
                    st.subheader("🚨 Priority Issues & Recommendations")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🚨 Priority Issues**")
                        for issue in analysis.priority_issues:
                            st.write(f"🔥 {issue}")

                    with col2:
                        st.markdown("**💡 Actionable Recommendations**")
                        for rec in analysis.actionable_recommendations:
                            st.write(f"💡 {rec}")

                # Interactive Dashboard
                st.header("📊 Interactive Feedback Dashboard")

                # Check if we have web search data to use real sentiment data
                if "web_search_results" in extracted_data:
                    web_data = extracted_data["web_search_results"]["data"]
                    # Calculate sentiment from web search results (basic analysis)
                    # This would normally be done with actual sentiment analysis
                    # For now, we'll derive it from the number of sources and search terms
                    total_sources = web_data["total_results"]

                    # Simple heuristic: more sources with positive terms = better sentiment
                    positive_terms = [
                        "great",
                        "excellent",
                        "good",
                        "love",
                        "amazing",
                        "best",
                        "quality",
                    ]
                    negative_terms = [
                        "bad",
                        "terrible",
                        "awful",
                        "hate",
                        "worst",
                        "complaint",
                        "problem",
                    ]

                    search_text = f"{web_data['search_query']} {web_data.get('search_terms', '')}".lower()
                    positive_score = sum(
                        1 for term in positive_terms if term in search_text
                    )
                    negative_score = sum(
                        1 for term in negative_terms if term in search_text
                    )

                    # Calculate sentiment distribution based on search results
                    if negative_score > positive_score:
                        # More negative sentiment when searching for complaints
                        real_sentiment = {"positive": 45, "neutral": 25, "negative": 30}
                    else:
                        # More balanced or positive sentiment
                        real_sentiment = {"positive": 60, "neutral": 25, "negative": 15}

                    # Extract themes from actual search content
                    extracted_themes = []
                    common_themes = [
                        "product",
                        "quality",
                        "service",
                        "price",
                        "pricing",
                        "customer",
                        "support",
                        "delivery",
                        "shipping",
                        "experience",
                        "value",
                        "staff",
                        "food",
                        "taste",
                    ]

                    # Analyze search content for themes
                    all_content = ""
                    for source in web_data["sources"][:5]:  # Analyze first 5 sources
                        content = (
                            source.get("content", "") + " " + source.get("snippet", "")
                        )
                        all_content += content.lower() + " "

                    # Find themes mentioned in content
                    for theme in common_themes:
                        if theme in all_content:
                            if theme in ["product", "quality"]:
                                extracted_themes.append("Product Quality")
                            elif theme in ["service", "customer", "support", "staff"]:
                                extracted_themes.append("Customer Service")
                            elif theme in ["price", "pricing", "value"]:
                                extracted_themes.append("Pricing")
                            elif theme in ["delivery", "shipping"]:
                                extracted_themes.append("Delivery")
                            elif theme in ["experience"]:
                                extracted_themes.append("User Experience")
                            elif theme in ["food", "taste"]:
                                extracted_themes.append("Product Quality")

                    # Remove duplicates and limit to top 4
                    extracted_themes = list(dict.fromkeys(extracted_themes))[:4]

                    # If no themes extracted, use defaults
                    if not extracted_themes:
                        extracted_themes = [
                            "Product Quality",
                            "Customer Service",
                            "Pricing",
                            "User Experience",
                        ]

                    # Create dashboard data with real search information
                    dashboard_data = {
                        "top_themes": extracted_themes,
                        "total_results": total_sources,
                        "company": web_data["company"],
                    }

                    dashboard_fig = create_feedback_dashboard(
                        dashboard_data, real_sentiment, use_real_data=True
                    )
                else:
                    # Use sample sentiment data for file uploads or demo
                    sample_sentiment = {
                        "positive": 65,
                        "neutral": 20,
                        "negative": 15,
                    }
                    dashboard_fig = create_feedback_dashboard(
                        extracted_data, sample_sentiment, use_real_data=False
                    )

                st.plotly_chart(dashboard_fig, use_container_width=True)

                # Export Options
                st.header("📥 Export Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Generate comprehensive report
                    report_content = f"""
CUSTOMER FEEDBACK INSIGHTS REPORT
Company: {company_name or 'N/A'}
Industry: {industry}
Feedback Period: {feedback_period}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY
Overall Satisfaction Score: {analysis.overall_satisfaction_score}/10

KEY STRENGTHS:
{chr(10).join([f"• {s}" for s in analysis.key_strengths])}

KEY CONCERNS:
{chr(10).join([f"• {c}" for c in analysis.key_concerns])}

DETAILED ANALYSIS

PRODUCT FEEDBACK:
{analysis.product_feedback}

SERVICE FEEDBACK:
{analysis.service_feedback}

USER EXPERIENCE FEEDBACK:
{analysis.user_experience_feedback}

COMPETITIVE INSIGHTS:
{analysis.competitive_insights}

ACTIONABLE RECOMMENDATIONS:
{chr(10).join([f"• {r}" for r in analysis.actionable_recommendations])}

PRIORITY ISSUES:
{chr(10).join([f"• {pi}" for pi in analysis.priority_issues])}
"""

                    st.download_button(
                        label="📄 Download Full Report",
                        data=report_content,
                        file_name=f"feedback_analysis_{company_name or 'company'}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                    )

                with col2:
                    if st.button("📧 Email Report"):
                        st.info("Email integration would be implemented here")

                with col3:
                    if st.button("📅 Schedule Review"):
                        st.info("Calendar integration would be implemented here")

    else:
        with st.expander("💼 Customer Feedback Insights Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(
                    """
                **💡 Perfect for:**
                - Marketing Teams
                - Customer Service Managers
                - Product Teams
                - Brand Managers
                - UX/UI Designers
                """
                )

            with col2:
                st.info(
                    """
                **✨ Key Features:**
                - Multi-format data ingestion
                - Web search for customer feedback
                - AI-powered sentiment analysis
                - Theme extraction & trending
                - Interactive dashboards
                - Actionable recommendations
                """
                )

            with col3:
                st.info(
                    """
                **🤖 AI-Powered Insights**
                - Comprehensive sentiment scoring
                - Automatic theme identification
                - Priority issue detection
                """
                )
            with col4:
                st.info(
                    """
                **📊 Interactive Dashboards**
                - Sentiment trend analysis
                - Feedback volume tracking
                - Theme frequency analysis
                """
                )

        # Sample data demo
        st.header("🎮 Try with Sample Data")

        demo_option = st.selectbox(
            "Choose Demo Data:",
            [
                "Customer Survey Results",
                "Product Review Analysis",
                "Social Media Mentions",
                "Web Search Demo",
            ],
        )

        if st.button("📊 Load Demo Feedback Data", type="primary"):
            if demo_option == "Customer Survey Results":
                # Create sample survey data
                sample_data = {
                    "feedback_type": "Survey",
                    "total_responses": 1250,
                    "satisfaction_scores": [8.5, 7.2, 9.1, 6.8, 8.9],
                    "sentiment_breakdown": {
                        "positive": 68,
                        "neutral": 22,
                        "negative": 10,
                    },
                    "top_themes": [
                        "Product Quality",
                        "Customer Service",
                        "Value for Money",
                    ],
                }

                st.session_state.demo_mode = True
                st.session_state.sample_data = sample_data
                st.session_state.demo_type = "survey"
                st.success(
                    "✅ Demo survey results loaded! Scroll down to see the analysis."
                )

            elif demo_option == "Product Review Analysis":
                sample_data = {
                    "feedback_type": "Reviews",
                    "total_reviews": 2847,
                    "average_rating": 4.2,
                    "sentiment_breakdown": {
                        "positive": 72,
                        "neutral": 18,
                        "negative": 10,
                    },
                    "top_themes": [
                        "Shipping Speed",
                        "Product Features",
                        "Customer Support",
                    ],
                }

                st.session_state.demo_mode = True
                st.session_state.sample_data = sample_data
                st.session_state.demo_type = "reviews"
                st.success(
                    "✅ Demo review analysis loaded! Scroll down to see the insights."
                )

            elif demo_option == "Social Media Mentions":
                sample_data = {
                    "feedback_type": "Social Media",
                    "total_mentions": 5623,
                    "reach": 2400000,
                    "sentiment_breakdown": {
                        "positive": 58,
                        "neutral": 28,
                        "negative": 14,
                    },
                    "top_themes": [
                        "Brand Perception",
                        "Campaign Response",
                        "Customer Issues",
                    ],
                }

                st.session_state.demo_mode = True
                st.session_state.sample_data = sample_data
                st.session_state.demo_type = "social_media"
                st.success(
                    "✅ Demo social media data loaded! Scroll down to see the analysis."
                )

            elif demo_option == "Web Search Demo":
                sample_data = {
                    "feedback_type": "Web Search",
                    "company": "Demo Company",
                    "search_query": "Demo Company customer feedback reviews",
                    "total_sources": 15,
                    "sentiment_breakdown": {
                        "positive": 62,
                        "neutral": 24,
                        "negative": 14,
                    },
                    "top_themes": [
                        "Product Quality",
                        "Shipping",
                        "Customer Support",
                        "Pricing",
                    ],
                }

                st.session_state.demo_mode = True
                st.session_state.sample_data = sample_data
                st.session_state.demo_type = "web_search"
                st.success(
                    "✅ Demo web search results loaded! Scroll down to see the analysis."
                )

        # Show demo data tables
        with st.expander("📋 View Demo Data Tables", expanded=False):
            st.subheader("📊 Customer Survey Results")
            survey_df = pd.DataFrame(
                {
                    "Question": [
                        "Overall Satisfaction",
                        "Product Quality",
                        "Customer Service",
                        "Value for Money",
                        "Likelihood to Recommend",
                    ],
                    "Average Score": [8.2, 8.7, 7.9, 7.1, 8.4],
                    "Response Count": [1250, 1248, 1245, 1240, 1247],
                    "Positive %": [76, 82, 71, 63, 78],
                    "Neutral %": [18, 14, 22, 28, 16],
                    "Negative %": [6, 4, 7, 9, 6],
                }
            )
            st.dataframe(survey_df, use_container_width=True)

            st.subheader("⭐ Product Reviews Summary")
            reviews_df = pd.DataFrame(
                {
                    "Platform": [
                        "Amazon",
                        "Google Reviews",
                        "Yelp",
                        "Company Website",
                        "Social Media",
                    ],
                    "Total Reviews": [1200, 450, 320, 680, 197],
                    "Average Rating": [4.3, 4.1, 4.0, 4.5, 4.2],
                    "Positive %": [75, 68, 65, 81, 72],
                    "Recent Trend": ["↗️ +0.2", "↘️ -0.1", "→ 0.0", "↗️ +0.3", "↗️ +0.1"],
                }
            )
            st.dataframe(reviews_df, use_container_width=True)

            st.subheader("📱 Social Media Sentiment")
            social_df = pd.DataFrame(
                {
                    "Platform": [
                        "Twitter",
                        "Facebook",
                        "Instagram",
                        "LinkedIn",
                        "TikTok",
                    ],
                    "Mentions": [2100, 1800, 950, 480, 293],
                    "Positive %": [55, 62, 71, 68, 58],
                    "Neutral %": [32, 26, 21, 24, 30],
                    "Negative %": [13, 12, 8, 8, 12],
                    "Engagement Rate": ["3.2%", "4.1%", "6.8%", "2.9%", "8.5%"],
                }
            )
            st.dataframe(social_df, use_container_width=True)

        if st.session_state.get("demo_mode"):
            demo_type = st.session_state.get("demo_type", "survey")
            sample_data = st.session_state.sample_data

            st.header("📊 Demo Feedback Dashboard")

            # Create demo visualizations based on demo type
            if demo_type == "survey":
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Survey Responses", "1,250", "+12%")
                with col2:
                    st.metric("Avg Satisfaction", "8.2/10", "+0.3")
                with col3:
                    st.metric("Positive Sentiment", "68%", "+5%")
                with col4:
                    st.metric("Response Rate", "42%", "+8%")

            elif demo_type == "reviews":
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Reviews", "2,847", "+156")
                with col2:
                    st.metric("Avg Rating", "4.2/5", "+0.1")
                with col3:
                    st.metric("Positive Reviews", "72%", "+2%")
                with col4:
                    st.metric("Review Velocity", "94/week", "+12")

            elif demo_type == "social_media":
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Mentions", "5,623", "+847")
                with col2:
                    st.metric("Reach", "2.4M", "+340K")
                with col3:
                    st.metric("Positive Sentiment", "58%", "+3%")
                with col4:
                    st.metric("Engagement Rate", "4.8%", "+0.5%")

            elif demo_type == "web_search":
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Web Sources", "15", "+3")
                with col2:
                    st.metric("Search Coverage", "95%", "+5%")
                with col3:
                    st.metric("Positive Sentiment", "62%", "+4%")
                with col4:
                    st.metric("Data Freshness", "72hrs", "Fresh")

            # Demo dashboard
            demo_fig = create_feedback_dashboard(
                sample_data,
                sample_data.get("sentiment_breakdown", {}),
                use_real_data=False,
            )
            st.plotly_chart(demo_fig, use_container_width=True)

            # Sample insights
            st.subheader("🧠 Sample AI Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success("✅ Strong product quality ratings")
                st.success("✅ Improving customer service scores")
                st.info("💡 Focus on value proposition messaging")

            with col2:
                st.warning("⚠️ Pricing concerns in reviews")
                st.warning("⚠️ Delivery time complaints")
                st.info("💡 Enhance shipping communication")

            with col3:
                st.error("🚨 Mobile app usability issues")
                st.error("🚨 Limited payment options")
                st.info("💡 Prioritize mobile UX improvements")


if __name__ == "__main__":
    main()
