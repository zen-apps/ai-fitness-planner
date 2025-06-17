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

# Load environment variables
load_dotenv("./config/dev.env")


# Pydantic models for structured financial analysis
class FinancialRatios(BaseModel):
    """Key financial ratios and metrics."""

    current_ratio: Optional[float] = Field(
        description="Current assets / Current liabilities"
    )
    quick_ratio: Optional[float] = Field(
        description="Quick assets / Current liabilities"
    )
    debt_to_equity: Optional[float] = Field(description="Total debt / Total equity")
    gross_margin: Optional[float] = Field(description="Gross profit margin percentage")
    net_margin: Optional[float] = Field(description="Net profit margin percentage")
    roa: Optional[float] = Field(description="Return on assets percentage")
    roe: Optional[float] = Field(description="Return on equity percentage")
    revenue_growth: Optional[float] = Field(description="Revenue growth percentage")


class FinancialInsights(BaseModel):
    """AI-generated financial insights and analysis."""

    overall_health_score: int = Field(description="Overall financial health score 1-10")
    key_strengths: List[str] = Field(description="Major financial strengths")
    key_concerns: List[str] = Field(description="Areas of financial concern")
    profitability_analysis: str = Field(description="Analysis of profitability trends")
    liquidity_analysis: str = Field(description="Analysis of liquidity position")
    leverage_analysis: str = Field(description="Analysis of debt and leverage")
    growth_analysis: str = Field(description="Analysis of growth trends")
    industry_comparison: str = Field(
        description="How this compares to industry standards"
    )
    recommendations: List[str] = Field(
        description="Specific recommendations for improvement"
    )
    risk_factors: List[str] = Field(description="Key financial risk factors")


class TrendAnalysis(BaseModel):
    """Analysis of financial trends over time."""

    revenue_trend: str = Field(description="Analysis of revenue trends")
    expense_trend: str = Field(description="Analysis of expense trends")
    profitability_trend: str = Field(description="Analysis of profitability trends")
    cash_flow_trend: str = Field(description="Analysis of cash flow trends")
    key_trend_insights: List[str] = Field(
        description="Key insights from trend analysis"
    )


# Cached LLM initialization
@st.cache_resource
def get_financial_llm():
    """Initialize OpenAI LLM for financial analysis."""
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
    )


# Configure page
st.set_page_config(
    page_title="Financial Health Analyzer",
    page_icon="📊",
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
    """Extract financial data from PDF using hybrid approach: PyPDF2 + AI analysis."""
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
                "structured_financial_data": "",
                "success": False,
                "extraction_method": "Failed - No Text",
            }

        st.success(
            f"✅ Extracted text from {num_pages} pages ({len(extracted_text):,} characters)"
        )

        # Now use AI to analyze and structure the extracted text
        st.info("🤖 Analyzing financial data with AI...")
        llm = get_financial_llm()

        # Split text if it's too long for the LLM
        max_text_length = 12000  # Conservative limit for token count
        if len(extracted_text) > max_text_length:
            text_to_analyze = (
                extracted_text[:max_text_length] + "...\n[Text truncated for analysis]"
            )
            st.info(
                f"📝 Text truncated to {max_text_length:,} characters for Demo AI analysis"
            )
        else:
            text_to_analyze = extracted_text

        financial_extraction_prompt = f"""
        You are a financial document analyst. Analyze this extracted text from a financial PDF document and identify key financial information.
        
        EXTRACTED TEXT:
        {text_to_analyze}
        
        Please provide a structured analysis including:
        
        1. **Document Type**: What type of financial document is this? (Annual Report, 10-K, Earnings Report, etc.)
        
        2. **Key Financial Data Found**:
           - Income Statement items (revenue, expenses, net income, etc.)
           - Balance Sheet items (assets, liabilities, equity, etc.)
           - Cash Flow items (operating, investing, financing cash flows)
           - Financial ratios or percentages
           - Time periods covered
        
        3. **Key Financial Metrics** (extract specific numbers with context):
           - Revenue figures
           - Profit/Loss amounts
           - Asset values
           - Debt levels
           - Any growth rates or percentages
        
        4. **Summary**: Provide a brief summary of the company's financial position based on the available data.
        
        If this doesn't appear to be a financial document or contains no financial data, please state that clearly.
        
        Format your response in a clear, structured way that would be useful for financial analysis.
        """

        try:
            financial_response = llm.invoke(
                [HumanMessage(content=financial_extraction_prompt)]
            )
            structured_data = financial_response.content
            st.success("✅ AI analysis completed successfully")
        except Exception as ai_error:
            st.warning(f"AI analysis failed: {ai_error}")
            structured_data = "AI analysis failed. Raw text extraction was successful but could not be processed by AI."

        return {
            "text": extracted_text,
            "structured_financial_data": structured_data,
            "success": True,
            "text_length": len(extracted_text),
            "extraction_method": "PyPDF2 + AI Analysis",
            "pages_processed": num_pages,
        }

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return {
            "text": "",
            "structured_financial_data": "",
            "success": False,
            "extraction_method": "Failed",
        }


def extract_data_from_excel(excel_file):
    """Extract financial data from Excel file."""
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


def calculate_financial_ratios(financial_data):
    """Calculate key financial ratios from extracted data."""
    try:
        ratios = {}

        # This is a simplified example - in practice, you'd need to map
        # the extracted data to specific financial statement items
        if (
            "current_assets" in financial_data
            and "current_liabilities" in financial_data
        ):
            ratios["current_ratio"] = (
                financial_data["current_assets"] / financial_data["current_liabilities"]
            )

        if "total_debt" in financial_data and "total_equity" in financial_data:
            ratios["debt_to_equity"] = (
                financial_data["total_debt"] / financial_data["total_equity"]
            )

        if "gross_profit" in financial_data and "revenue" in financial_data:
            ratios["gross_margin"] = (
                financial_data["gross_profit"] / financial_data["revenue"]
            ) * 100

        if "net_income" in financial_data and "revenue" in financial_data:
            ratios["net_margin"] = (
                financial_data["net_income"] / financial_data["revenue"]
            ) * 100

        return ratios
    except Exception as e:
        st.warning(f"Could not calculate all financial ratios: {e}")
        return {}


def generate_financial_analysis(financial_data, company_name="Company"):
    """Generate AI-powered financial analysis."""
    llm = get_financial_llm()

    # Prepare data summary for LLM
    data_summary = f"""
    Financial Analysis for: {company_name}
    
    Extracted Financial Data:
    {str(financial_data)[:2000]}
    
    Please analyze this financial information and provide:
    1. Overall financial health assessment (score 1-10)
    2. Key strengths and areas of concern
    3. Profitability, liquidity, and leverage analysis
    4. Growth analysis and trends
    5. Industry comparison insights
    6. Specific recommendations for improvement
    7. Key risk factors to monitor
    
    Focus on actionable insights that would be valuable for CFOs and financial decision-makers.
    """

    try:
        structured_llm = llm.with_structured_output(FinancialInsights)
        result = structured_llm.invoke([HumanMessage(content=data_summary)])
        return result
    except Exception as e:
        st.error(f"Error generating financial analysis: {e}")
        return None


def generate_trend_analysis(time_series_data):
    """Generate trend analysis from time series financial data."""
    llm = get_financial_llm()

    trend_prompt = f"""
    Analyze the following time series financial data for trends:
    
    {str(time_series_data)[:1500]}
    
    Provide analysis of:
    1. Revenue trends over time
    2. Expense patterns and trends
    3. Profitability trends
    4. Cash flow trends
    5. Key insights from the trend analysis
    
    Focus on identifying patterns, seasonal variations, and concerning trends.
    """

    try:
        structured_llm = llm.with_structured_output(TrendAnalysis)
        result = structured_llm.invoke([HumanMessage(content=trend_prompt)])
        return result
    except Exception as e:
        st.error(f"Error generating trend analysis: {e}")
        return None


def create_financial_dashboard(data, ratios):
    """Create interactive financial dashboard with multiple charts."""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Financial Ratios",
            "Revenue vs Expenses",
            "Profitability Trends",
            "Key Metrics",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "indicator"}],
        ],
    )

    # Financial Ratios Bar Chart
    if ratios:
        ratio_names = list(ratios.keys())
        ratio_values = list(ratios.values())

        fig.add_trace(
            go.Bar(x=ratio_names, y=ratio_values, name="Financial Ratios"), row=1, col=1
        )

    # Sample data visualization (in practice, use extracted data)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    revenue = [100, 120, 110, 130, 140, 150]
    expenses = [80, 95, 85, 100, 105, 110]

    fig.add_trace(
        go.Scatter(x=months, y=revenue, mode="lines+markers", name="Revenue"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=months, y=expenses, mode="lines+markers", name="Expenses"),
        row=1,
        col=2,
    )

    # Profitability trend
    profit_margin = [(r - e) / r * 100 for r, e in zip(revenue, expenses)]
    fig.add_trace(
        go.Scatter(
            x=months, y=profit_margin, mode="lines+markers", name="Profit Margin %"
        ),
        row=2,
        col=1,
    )

    # Key metric indicator
    current_margin = profit_margin[-1] if profit_margin else 0
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_margin,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Current Profit Margin %"},
            delta={"reference": 20},
            gauge={
                "axis": {"range": [None, 50]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 10], "color": "lightgray"},
                    {"range": [10, 25], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 20,
                },
            },
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=True, title_text="Financial Health Dashboard"
    )
    return fig


def main():
    """Main application function."""

    # Initialize session state
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False
    if "demo_type" not in st.session_state:
        st.session_state.demo_type = None

    st.title("📊 Financial Health Analyzer")
    st.markdown(
        "*Upload financial statements and get AI-powered insights with interactive dashboards*"
    )

    # Sidebar for file upload and options
    with st.sidebar:
        st.header("📁 Upload Financial Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF or Excel files",
            type=["pdf", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload financial statements, income statements, balance sheets, or cash flow statements",
        )

        st.markdown("---")

        # Company information
        st.header("🏢 Company Information")
        company_name = st.text_input("Company Name", placeholder="Enter company name")
        industry = st.selectbox(
            "Industry",
            [
                "Technology",
                "Healthcare",
                "Manufacturing",
                "Retail",
                "Finance",
                "Energy",
                "Other",
            ],
        )

        analysis_period = st.selectbox(
            "Analysis Period",
            ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Annual 2023", "Custom"],
        )

        st.markdown("---")

        # Analysis options
        st.header("⚙️ Analysis Options")
        include_trends = st.checkbox("Include Trend Analysis", value=True)
        include_ratios = st.checkbox("Calculate Financial Ratios", value=True)
        include_benchmarks = st.checkbox("Industry Benchmarking", value=True)

        st.markdown("---")

        # API Status
        st.header("🔌 API Status")
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI Connected")
        else:
            st.error("❌ OpenAI API Key Missing")

    # Main content area
    if uploaded_files:
        st.header("📋 Document Processing Results")

        extracted_data = {}
        all_dataframes = []

        # Process each uploaded file
        for file in uploaded_files:
            st.subheader(f"📄 Processing: {file.name}")

            if file.type == "application/pdf":
                with st.spinner(f"Extracting data from {file.name}..."):
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
                                "Text Length", f"{result.get('text_length', 0):,} chars"
                            )
                        with col3:
                            st.metric(
                                "Status",
                                "✅ Success" if result["success"] else "❌ Failed",
                            )

                        # Display structured financial data
                        if result.get("structured_financial_data"):
                            with st.expander("📊 Structured Financial Data"):
                                st.write(result["structured_financial_data"])

                        # Display extracted text preview
                        if result["text"]:
                            with st.expander("📝 Extracted Text Preview"):
                                preview_text = (
                                    result["text"][:2000] + "..."
                                    if len(result["text"]) > 2000
                                    else result["text"]
                                )
                                st.text_area("Text Content", preview_text, height=300)

            elif file.type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ]:
                with st.spinner(f"Reading Excel file {file.name}..."):
                    result = extract_data_from_excel(file)
                    if result["success"]:
                        extracted_data[file.name] = result

                        st.success(f"✅ Read {len(result['sheets'])} sheets from Excel")
                        for sheet_info in result["sheets"]:
                            with st.expander(
                                f"Sheet: {sheet_info['sheet_name']} - Shape: {sheet_info['shape']}"
                            ):
                                st.dataframe(sheet_info["data"])
                                all_dataframes.append(sheet_info["data"])

        # Financial Analysis Section
        if extracted_data:
            st.header("🧠 AI Financial Analysis")

            if st.button("🚀 Generate Financial Analysis", type="primary"):
                with st.spinner("Analyzing financial data with AI..."):
                    # Combine all extracted data for analysis
                    combined_data = {
                        "files": extracted_data,
                        "company": company_name,
                        "industry": industry,
                        "period": analysis_period,
                    }

                    analysis = generate_financial_analysis(
                        combined_data, company_name or "Your Company"
                    )

                    if analysis:
                        st.session_state.financial_analysis = analysis
                        st.success("✅ Financial analysis completed!")

            # Display Analysis Results
            if "financial_analysis" in st.session_state:
                analysis = st.session_state.financial_analysis

                # Key Metrics Overview
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    score_color = (
                        "green"
                        if analysis.overall_health_score >= 7
                        else "orange" if analysis.overall_health_score >= 5 else "red"
                    )
                    st.metric(
                        "Financial Health Score", f"{analysis.overall_health_score}/10"
                    )

                with col2:
                    st.metric("Key Strengths", len(analysis.key_strengths))

                with col3:
                    st.metric("Areas of Concern", len(analysis.key_concerns))

                with col4:
                    st.metric("Risk Factors", len(analysis.risk_factors))

                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "📊 Overview",
                        "💹 Profitability",
                        "💰 Liquidity",
                        "📈 Growth",
                        "⚠️ Risks",
                    ]
                )

                with tab1:
                    st.subheader("🎯 Overall Financial Health")

                    health_color = (
                        "green"
                        if analysis.overall_health_score >= 7
                        else "orange" if analysis.overall_health_score >= 5 else "red"
                    )
                    st.markdown(
                        f"**Health Score:** :{health_color}[{analysis.overall_health_score}/10]"
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

                    st.markdown("**🏭 Industry Comparison**")
                    st.info(analysis.industry_comparison)

                with tab2:
                    st.subheader("💹 Profitability Analysis")
                    st.write(analysis.profitability_analysis)

                    # Sample profitability chart
                    fig_profit = px.bar(
                        x=["Gross Margin", "Operating Margin", "Net Margin"],
                        y=[25, 15, 8],  # Sample data
                        title="Profitability Margins",
                        color_discrete_sequence=["#1f77b4"],
                    )
                    st.plotly_chart(fig_profit, use_container_width=True)

                with tab3:
                    st.subheader("💰 Liquidity Analysis")
                    st.write(analysis.liquidity_analysis)

                    # Sample liquidity ratios
                    liquidity_data = {
                        "Ratio": ["Current Ratio", "Quick Ratio", "Cash Ratio"],
                        "Value": [2.1, 1.5, 0.8],
                        "Benchmark": [2.0, 1.0, 0.5],
                    }
                    fig_liquidity = px.bar(
                        pd.DataFrame(liquidity_data),
                        x="Ratio",
                        y=["Value", "Benchmark"],
                        title="Liquidity Ratios vs Benchmarks",
                        barmode="group",
                    )
                    st.plotly_chart(fig_liquidity, use_container_width=True)

                with tab4:
                    st.subheader("📈 Growth Analysis")
                    st.write(analysis.growth_analysis)

                    # Sample growth trends
                    growth_data = {
                        "Period": ["Q1", "Q2", "Q3", "Q4"],
                        "Revenue Growth": [5, 8, 12, 15],
                        "Profit Growth": [3, 6, 10, 18],
                    }
                    fig_growth = px.line(
                        pd.DataFrame(growth_data),
                        x="Period",
                        y=["Revenue Growth", "Profit Growth"],
                        title="Growth Trends (%)",
                        markers=True,
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)

                with tab5:
                    st.subheader("⚠️ Risk Assessment")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🚨 Key Risk Factors**")
                        for risk in analysis.risk_factors:
                            st.write(f"⚠️ {risk}")

                    with col2:
                        st.markdown("**💡 Recommendations**")
                        for rec in analysis.recommendations:
                            st.write(f"💡 {rec}")

                # Interactive Dashboard
                st.header("📊 Interactive Financial Dashboard")

                # Calculate sample ratios for dashboard
                sample_ratios = {
                    "Current Ratio": 2.1,
                    "Debt/Equity": 0.4,
                    "ROA": 8.5,
                    "ROE": 15.2,
                    "Gross Margin": 35.0,
                }

                dashboard_fig = create_financial_dashboard(
                    extracted_data, sample_ratios
                )
                st.plotly_chart(dashboard_fig, use_container_width=True)

                # Export Options
                st.header("📥 Export Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Generate comprehensive report
                    report_content = f"""
FINANCIAL HEALTH ANALYSIS REPORT
Company: {company_name or 'N/A'}
Industry: {industry}
Analysis Period: {analysis_period}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY
Overall Health Score: {analysis.overall_health_score}/10

KEY STRENGTHS:
{chr(10).join([f"• {s}" for s in analysis.key_strengths])}

KEY CONCERNS:
{chr(10).join([f"• {c}" for c in analysis.key_concerns])}

DETAILED ANALYSIS

PROFITABILITY:
{analysis.profitability_analysis}

LIQUIDITY:
{analysis.liquidity_analysis}

LEVERAGE:
{analysis.leverage_analysis}

GROWTH:
{analysis.growth_analysis}

INDUSTRY COMPARISON:
{analysis.industry_comparison}

RECOMMENDATIONS:
{chr(10).join([f"• {r}" for r in analysis.recommendations])}

RISK FACTORS:
{chr(10).join([f"• {rf}" for rf in analysis.risk_factors])}
"""

                    st.download_button(
                        label="📄 Download Full Report",
                        data=report_content,
                        file_name=f"financial_analysis_{company_name or 'company'}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                    )

                with col2:
                    if st.button("📧 Email Report"):
                        st.info("Email integration would be implemented here")

                with col3:
                    if st.button("📋 Schedule Follow-up"):
                        st.info("Calendar integration would be implemented here")

    else:
        with st.expander("📊 Financial Health Analyzer Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(
                    """
                **💡 Perfect for:**
                - CFOs and Finance Teams
                - Investment Analysts
                - Board Presentations
                - Due Diligence
                - Performance Reviews
                """
                )

            with col2:
                st.info(
                    """
                **✨ Key Features:**
                - Upload & analyze in minutes
                - AI-powered insights
                - Interactive dashboards  
                - Professional reports
                - Industry benchmarks
                """
                )

            with col3:
                st.info(
                    """
                **🤖 AI-Powered Insights**
                - Comprehensive financial health scoring
                - Industry benchmarking
                - Risk assessment and recommendations
                """
                )
            with col4:
                st.info(
                    """
                **📊 Interactive Dashboards**
                - Trend analysis and forecasting
                - Comparative visualizations
                - Drill-down capabilities
                """
                )

        # Sample data demo
        st.header("🎮 Try with Sample Data")

        demo_option = st.selectbox(
            "Choose Demo Data:",
            [
                "Financial Statements (Income Statement, Balance Sheet, Cash Flow)",
                "Target Corp Annual Report 2024",
            ],
        )

        if st.button("📊 Load Demo Financial Data", type="primary"):
            if (
                demo_option
                == "Financial Statements (Income Statement, Balance Sheet, Cash Flow)"
            ):
                # Create sample financial data
                sample_data = {
                    "Revenue": [1000000, 1200000, 1100000, 1300000],
                    "Expenses": [800000, 950000, 880000, 1040000],
                    "Net_Income": [200000, 250000, 220000, 260000],
                    "Assets": [2000000, 2200000, 2100000, 2300000],
                    "Liabilities": [800000, 850000, 820000, 900000],
                }

                st.session_state.demo_mode = True
                st.session_state.sample_data = sample_data
                st.session_state.demo_type = "financial_statements"
                st.success(
                    "✅ Demo financial statements loaded! Scroll down to see the analysis."
                )

            elif demo_option == "Target Corp Annual Report 2024":
                # Load Target Corp annual report
                target_pdf_path = "./streamlit/data/financial_health/2024-Annual-Report-Target-Corporation.pdf"
                if os.path.exists(target_pdf_path):
                    try:
                        with open(target_pdf_path, "rb") as f:
                            # Create a file-like object for the extract function
                            class FileWrapper:
                                def __init__(self, file_content):
                                    self.content = file_content

                                def getvalue(self):
                                    return self.content

                            file_wrapper = FileWrapper(f.read())

                            with st.spinner(
                                "Extracting Target Corp financial data using AI..."
                            ):
                                result = extract_data_from_pdf(file_wrapper)

                            if result["success"]:
                                st.session_state.demo_mode = True
                                st.session_state.target_report_data = result
                                st.session_state.demo_type = "target_report"
                                st.success(
                                    "✅ Target Corp Annual Report processed with AI! Scroll down to see the analysis."
                                )
                                st.info(
                                    f"📊 Extracted {result.get('text_length', 0):,} characters of financial data"
                                )
                            else:
                                st.error(
                                    "Failed to extract data from Target Corp report"
                                )
                    except Exception as e:
                        st.error(f"Error loading Target Corp report: {e}")
                        st.info(
                            "Make sure OpenAI API key is configured for AI extraction"
                        )
                else:
                    st.error(
                        "Target Corp Annual Report not found. Please ensure the file is in ./data/financial_health/"
                    )
                    st.info(
                        "Expected path: ./data/financial_health/2024-Annual-Report-Target-Corporation.pdf"
                    )

        # Show demo dataframes in expandable section
        with st.expander("📋 View Demo Data Tables", expanded=False):
            st.subheader("💰 Income Statement (Quarterly)")
            income_df = pd.DataFrame(
                {
                    "Period": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
                    "Revenue": [1000000, 1200000, 1100000, 1300000],
                    "Cost of Goods Sold": [600000, 720000, 660000, 780000],
                    "Gross Profit": [400000, 480000, 440000, 520000],
                    "Operating Expenses": [200000, 230000, 220000, 260000],
                    "Operating Income": [200000, 250000, 220000, 260000],
                    "Interest Expense": [15000, 18000, 16000, 20000],
                    "Net Income": [185000, 232000, 204000, 240000],
                }
            )
            st.dataframe(income_df, use_container_width=True)

            st.subheader("🏦 Balance Sheet (Quarterly)")
            balance_df = pd.DataFrame(
                {
                    "Period": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
                    "Current Assets": [800000, 900000, 850000, 950000],
                    "Fixed Assets": [1200000, 1300000, 1250000, 1350000],
                    "Total Assets": [2000000, 2200000, 2100000, 2300000],
                    "Current Liabilities": [400000, 450000, 425000, 475000],
                    "Long-term Debt": [400000, 400000, 395000, 425000],
                    "Total Liabilities": [800000, 850000, 820000, 900000],
                    "Shareholders Equity": [1200000, 1350000, 1280000, 1400000],
                }
            )
            st.dataframe(balance_df, use_container_width=True)

            st.subheader("💸 Cash Flow Statement (Quarterly)")
            cashflow_df = pd.DataFrame(
                {
                    "Period": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
                    "Operating Cash Flow": [220000, 280000, 250000, 300000],
                    "Investing Cash Flow": [-50000, -80000, -45000, -100000],
                    "Financing Cash Flow": [-30000, -40000, -35000, -45000],
                    "Net Cash Flow": [140000, 160000, 170000, 155000],
                    "Cash Beginning": [300000, 440000, 600000, 770000],
                    "Cash Ending": [440000, 600000, 770000, 925000],
                }
            )
            st.dataframe(cashflow_df, use_container_width=True)

            st.subheader("📊 Key Financial Ratios")
            ratios_df = pd.DataFrame(
                {
                    "Ratio": [
                        "Current Ratio",
                        "Quick Ratio",
                        "Debt-to-Equity",
                        "Gross Margin %",
                        "Net Margin %",
                        "ROA %",
                        "ROE %",
                    ],
                    "Q1 2024": [2.0, 1.5, 0.67, 40.0, 18.5, 9.3, 15.4],
                    "Q2 2024": [2.0, 1.4, 0.63, 40.0, 19.3, 10.5, 17.2],
                    "Q3 2024": [2.0, 1.6, 0.64, 40.0, 18.5, 9.7, 15.9],
                    "Q4 2024": [2.0, 1.5, 0.64, 40.0, 18.5, 10.4, 17.1],
                    "Industry Avg": [1.8, 1.2, 0.75, 35.0, 15.0, 8.5, 14.0],
                }
            )
            st.dataframe(ratios_df, use_container_width=True)

        if st.session_state.get("demo_mode"):
            demo_type = st.session_state.get("demo_type", "financial_statements")

            if demo_type == "target_report":
                st.header("📊 Target Corp Annual Report Analysis")

                if "target_report_data" in st.session_state:
                    report_data = st.session_state.target_report_data

                    # Display extracted content
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Text Length",
                            f"{report_data.get('text_length', 0):,} chars",
                        )
                    with col2:
                        st.metric(
                            "Extraction Method",
                            report_data.get("extraction_method", "AI Base64"),
                        )
                    with col3:
                        st.metric("Processing Status", "✅ Complete")

                    # Show structured financial data
                    if report_data.get("structured_financial_data"):
                        with st.expander(
                            "📊 AI-Structured Financial Data", expanded=True
                        ):
                            st.write(report_data["structured_financial_data"])

                    # Show extracted text preview
                    if report_data.get("text"):
                        with st.expander("📄 Full Extracted Text", expanded=False):
                            preview_text = (
                                report_data["text"][:8000] + "..."
                                if len(report_data["text"]) > 8000
                                else report_data["text"]
                            )
                            st.text_area("Text Content", preview_text, height=400)

                    # Generate AI analysis for Target Corp
                    if st.button("🧠 Analyze Target Corp Financials", type="primary"):
                        with st.spinner(
                            "Analyzing Target Corp financial data with AI..."
                        ):
                            analysis = generate_financial_analysis(
                                {"target_report": report_data}, "Target Corporation"
                            )
                            if analysis:
                                st.session_state.financial_analysis = analysis
                                st.success(
                                    "✅ Target Corp financial analysis completed!"
                                )

            else:
                st.header("📊 Demo Financial Dashboard")

                sample_data = st.session_state.sample_data

                # Create demo visualizations
                periods = ["Q1", "Q2", "Q3", "Q4"]

                fig_demo = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "Revenue & Expenses",
                        "Profitability",
                        "Asset Growth",
                        "Key Ratios",
                    ),
                )

                # Revenue and expenses
                fig_demo.add_trace(
                    go.Bar(
                        x=periods,
                        y=sample_data["Revenue"],
                        name="Revenue",
                        marker_color="blue",
                    ),
                    row=1,
                    col=1,
                )
                fig_demo.add_trace(
                    go.Bar(
                        x=periods,
                        y=sample_data["Expenses"],
                        name="Expenses",
                        marker_color="red",
                    ),
                    row=1,
                    col=1,
                )

                # Profitability
                margins = [
                    (r - e) / r * 100
                    for r, e in zip(sample_data["Revenue"], sample_data["Expenses"])
                ]
                fig_demo.add_trace(
                    go.Scatter(
                        x=periods,
                        y=margins,
                        mode="lines+markers",
                        name="Profit Margin %",
                    ),
                    row=1,
                    col=2,
                )

                # Assets
                fig_demo.add_trace(
                    go.Scatter(
                        x=periods,
                        y=sample_data["Assets"],
                        mode="lines+markers",
                        name="Total Assets",
                    ),
                    row=2,
                    col=1,
                )

                # Ratios
                ratios = ["Current", "Debt/Equity", "ROA", "ROE"]
                ratio_values = [2.1, 0.4, 8.5, 15.2]
                fig_demo.add_trace(
                    go.Bar(x=ratios, y=ratio_values, name="Financial Ratios"),
                    row=2,
                    col=2,
                )

                fig_demo.update_layout(
                    height=600, showlegend=True, title_text="Sample Financial Dashboard"
                )
                st.plotly_chart(fig_demo, use_container_width=True)

                # Demo insights
                st.subheader("🧠 Sample AI Insights")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Health Score", "8/10", "+1")
                    st.success("✅ Strong liquidity position")
                    st.success("✅ Consistent revenue growth")

                with col2:
                    st.metric("Profit Margin", "18.5%", "+2.1%")
                    st.warning("⚠️ Rising expense ratio")
                    st.info("💡 Consider cost optimization")

                with col3:
                    st.metric("Growth Rate", "12%", "+3%")
                    st.success("✅ Outperforming industry")
                    st.info("💡 Expand market presence")


if __name__ == "__main__":
    main()
