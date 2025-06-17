import pandas as pd
from gtts import gTTS
import tempfile
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Load environment variables
load_dotenv("./config/dev.env")


# Pydantic models for structured outputs
class ContactInsights(BaseModel):
    """Structured insights about a contact."""

    company_overview: str = Field(description="Brief overview of the contact's company")
    recent_news: List[str] = Field(description="List of recent news about the company")
    business_opportunities: List[str] = Field(
        description="Potential business opportunities"
    )
    engagement_strategy: List[str] = Field(
        description="Recommended engagement strategies"
    )
    risk_factors: List[str] = Field(description="Potential risks or concerns")
    contact_score: int = Field(description="Contact quality score from 1-10")
    next_actions: List[str] = Field(description="Recommended next steps")


class AccountAnalysis(BaseModel):
    """Structured analysis of an account."""

    company_health: str = Field(description="Overall company health assessment")
    market_position: str = Field(description="Market position analysis")
    growth_potential: int = Field(description="Growth potential score 1-10")
    competitive_landscape: List[str] = Field(description="Key competitors")
    expansion_opportunities: List[str] = Field(
        description="Opportunities for account expansion"
    )
    retention_risks: List[str] = Field(description="Potential retention risks")


class OpportunityIntelligence(BaseModel):
    """AI insights for opportunities."""

    win_probability: int = Field(description="Estimated win probability percentage")
    key_success_factors: List[str] = Field(description="Critical success factors")
    competitive_threats: List[str] = Field(description="Competitive threats")
    recommended_actions: List[str] = Field(
        description="Recommended actions to increase win rate"
    )
    timeline_analysis: str = Field(description="Analysis of opportunity timeline")


class ContactSalesScript(BaseModel):
    """Structured sales script for a contact."""

    opening_statement: str = Field(description="Personalized opening statement")
    value_proposition: str = Field(description="Tailored value proposition")
    discovery_questions: List[str] = Field(description="Strategic discovery questions")
    objection_handling: List[str] = Field(
        description="Potential objections and responses"
    )
    closing_statement: str = Field(description="Compelling closing statement")
    call_to_action: str = Field(description="Specific next steps")
    script_tone: str = Field(description="Recommended tone for the conversation")
    key_talking_points: List[str] = Field(description="Important points to emphasize")
    script_effectiveness_score: int = Field(
        description="Estimated script effectiveness 1-10", default=8
    )


# Tool initialization functions
@st.cache_resource
def get_llm():
    """Initialize OpenAI LLM with caching."""
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
    )


@st.cache_resource
def get_search_tool(time_back_search=None):
    """Initialize search tool with caching."""
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

    tool = TavilySearch(
        max_results=5,
        topic="general",
        time_range=time_back,
    )
    return tool


# AI Analysis Functions
def generate_contact_insights(
    contact_name, company_name, email=None, time_back_search="1 month"
):
    """Generate AI insights for a contact using web search."""
    llm = get_llm()
    search = get_search_tool(time_back_search=time_back_search)

    # Search for company information
    search_query = (
        f"{company_name} recent news business updates. Include Date and Source."
    )
    search_results = search.invoke(search_query)

    prompt = f"""
    Analyze the following contact and company information to provide business insights:
    
    Contact: {contact_name}
    Company: {company_name}
    Email: {email or 'Not provided'}
    
    Search Results:
    {str(search_results)[:2000]}
    
    Provide structured insights including:
    - Company overview and recent developments
    - Business opportunities for engagement
    - Recommended engagement strategies
    - Risk factors to consider
    - Quality score for this contact (1-10)
    - Specific next actions to take
    
    Be concise and actionable in your recommendations.
    For Search Results, focus on the most relevant and recent information and always include the Date and Source of the information.
    """

    structured_llm = llm.with_structured_output(ContactInsights)
    return structured_llm.invoke([HumanMessage(content=prompt)])


def generate_account_analysis(
    account_name, account_data=None, time_back_search="1 month"
):
    """Generate comprehensive account analysis."""
    llm = get_llm()
    search = get_search_tool(time_back_search=time_back_search)

    # Search for account information
    search_results = search.invoke(
        f"{account_name} company analysis financial performance. Include Date and Source."
    )

    prompt = f"""
    Analyze this account for comprehensive business intelligence:
    
    Account: {account_name}
    Additional Data: {str(account_data)[:500] if account_data else 'None'}
    
    Market Research:
    {str(search_results)[:2000]}
    
    Provide analysis covering:
    - Overall company health and stability
    - Market position relative to competitors
    - Growth potential score (1-10)
    - Key competitors in their space
    - Opportunities for account expansion
    - Potential retention risks
    
    Focus on actionable business intelligence.
    For Market Research, always include the Date and Source of the information.
    """

    structured_llm = llm.with_structured_output(AccountAnalysis)
    return structured_llm.invoke([HumanMessage(content=prompt)])


def generate_contact_sales_script(
    contact_name,
    contact_insights,
    script_style="Professional",
    script_length="Medium (10-15 min call)",
    contact_details=None,
):
    """Generate AI-powered sales script for a contact."""
    llm = get_llm()

    # Extract length guidance
    length_guidance = {
        "Short (5-7 min call)": "Keep it concise and focused on key value props. 2-3 discovery questions maximum.",
        "Medium (10-15 min call)": "Include 4-5 discovery questions and comprehensive objection handling.",
        "Long (20-30 min call)": "Comprehensive discussion with detailed exploration and relationship building.",
    }

    # Extract style guidance
    style_guidance = {
        "Professional": "formal, respectful, and business-focused",
        "Consultative": "advisory, question-heavy, and solution-oriented",
        "Relationship-Building": "warm, personal, and trust-focused",
        "Direct": "straightforward, efficient, and results-oriented",
        "Technical": "detailed, data-driven, and feature-focused",
    }

    prompt = f"""
    Create a personalized sales script for the following contact:
    
    Contact: {contact_name}
    Company: {contact_details.get('Account_Name', 'Unknown') if contact_details else 'Unknown'}
    Title: {contact_details.get('Title', 'Unknown') if contact_details else 'Unknown'}
    Department: {contact_details.get('Department', 'Unknown') if contact_details else 'Unknown'}
    
    Script Requirements:
    - Style: {script_style} ({style_guidance.get(script_style, 'professional and engaging')})
    - Length: {script_length} - {length_guidance.get(script_length, 'Balanced approach with key elements')}
    
    Contact AI Insights:
    - Company Overview: {contact_insights.company_overview}
    - Contact Quality Score: {contact_insights.contact_score}/10
    - Business Opportunities: {', '.join(contact_insights.business_opportunities) if contact_insights.business_opportunities else 'General business growth'}
    - Engagement Strategy: {', '.join(contact_insights.engagement_strategy) if contact_insights.engagement_strategy else 'Standard professional approach'}
    - Risk Factors: {', '.join(contact_insights.risk_factors) if contact_insights.risk_factors else 'No major risks identified'}
    - Recent News: {', '.join(contact_insights.recent_news) if contact_insights.recent_news else 'No recent news available'}
    
    Generate a comprehensive sales script that includes:
    
    1. OPENING STATEMENT (30-45 seconds):
       - Reference their specific company and role
       - Mention relevant recent news or industry trends
       - Clear reason for the call
    
    2. VALUE PROPOSITION (1-2 minutes):
       - Tailored to their identified business opportunities
       - Address their company's specific challenges
       - Clear differentiation from competitors
    
    3. DISCOVERY QUESTIONS:
       - {length_guidance.get(script_length, '3-4')} strategic questions
       - Focus on uncovering pain points and priorities
       - Build on the AI insights provided
    
    4. OBJECTION HANDLING:
       - Address the identified risk factors
       - 3-4 common objections with responses
       - Proactive handling of potential concerns
    
    5. CLOSING STATEMENT:
       - Summarize value discussed
       - Create urgency or importance
       - Natural transition to call-to-action
    
    6. CALL TO ACTION:
       - Specific, clear next step
       - Include timing and logistics
       - Easy for prospect to say yes
    
    7. CONVERSATION TONE:
       - Should be {style_guidance.get(script_style, 'professional and engaging')}
       - Match the {script_style.lower()} approach
    
    8. KEY TALKING POINTS:
       - 5-7 critical points to emphasize
       - Based on AI insights and opportunities
       - Memorable and impactful messages
    
    9. EFFECTIVENESS SCORE:
       - Rate this script's potential effectiveness 1-10
       - Based on personalization and insight quality
    
    Make the script conversational, natural, and highly personalized to {contact_name} at {contact_details.get('Account_Name', 'their company') if contact_details else 'their company'}.
    Ensure it flows naturally and doesn't sound robotic or overly scripted.
    """

    try:
        structured_llm = llm.with_structured_output(ContactSalesScript)
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        return result
    except Exception as e:
        st.error(f"Error in LLM call: {str(e)}")
        # Return a fallback script structure
        return ContactSalesScript(
            opening_statement=f"Hi {contact_name}, I hope you're doing well. I wanted to reach out because I've been following {contact_details.get('Account_Name', 'your company') if contact_details else 'your company'} and I believe we might be able to help with some growth opportunities.",
            value_proposition="Based on what I understand about your business, we specialize in helping companies like yours achieve their growth objectives through innovative solutions.",
            discovery_questions=[
                "What are your biggest challenges in your current role?",
                "How are you currently handling [relevant business area]?",
                "What would success look like for you in the next quarter?",
            ],
            objection_handling=[
                "If timing is a concern: I understand timing is important. Would it help if we could show quick wins within the first 30 days?",
                "If budget is an issue: Let's focus on the ROI. What if this solution could pay for itself within 6 months?",
            ],
            closing_statement="Based on our conversation, it sounds like there's a real opportunity to help you achieve your goals.",
            call_to_action="Would you be open to a 15-minute demo next week to see how this could work for your specific situation?",
            script_tone=f"{script_style} and conversational",
            key_talking_points=[
                "Focus on their specific industry challenges",
                "Emphasize quick time to value",
                "Highlight relevant case studies",
                "Address their growth objectives",
            ],
            script_effectiveness_score=7,
        )


# Text-to-Speech Functions
def generate_script_audio(script_text, language_code="en", tld="us", slow=False):
    """Generate TTS audio for sales script."""
    try:
        # Clean up the script text for better TTS
        cleaned_text = script_text.replace("\n", " ").replace("  ", " ").strip()

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()

        # Generate TTS
        if language_code == "en":
            tts = gTTS(text=cleaned_text, lang=language_code, tld=tld, slow=slow)
        else:
            tts = gTTS(text=cleaned_text, lang=language_code, slow=slow)

        tts.save(temp_file.name)

        # Read the file and return bytes
        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()

        # Clean up temporary file
        os.unlink(temp_file.name)
        return audio_bytes

    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")
        st.info("Please check your internet connection and try again.")
        return None


def format_script_for_audio(script):
    """Format the sales script for better TTS pronunciation."""
    audio_script = f"""
    This is your personalized sales script.
    
    Opening Statement: {script.opening_statement}
    
    Value Proposition: {script.value_proposition}
    
    Discovery Questions: 
    """

    for i, question in enumerate(script.discovery_questions, 1):
        audio_script += f"Question {i}: {question}. "

    audio_script += f"""
    
    Closing Statement: {script.closing_statement}
    
    Call to Action: {script.call_to_action}
    
    Remember to maintain a {script.script_tone.lower()} tone throughout your conversation.
    """

    return audio_script


# Additional AI helper functions for future features
def generate_opportunity_intelligence(
    opportunity_name, opportunity_data=None, account_context=None
):
    """Generate AI insights for opportunities (future feature)."""
    # This function can be implemented for the Opportunity Intelligence page
    pass


def generate_email_template(contact_insights, script_context=None):
    """Generate personalized email templates based on insights."""
    # This function can be used to create more sophisticated email templates
    pass


def analyze_pipeline_health(opportunities_df):
    """Use AI to analyze pipeline health and provide recommendations."""
    # This function can provide AI-driven pipeline analysis
    pass


# Prompt templates for consistency
CONTACT_INSIGHTS_PROMPT_TEMPLATE = """
Analyze the following contact and company information to provide business insights:

Contact: {contact_name}
Company: {company_name}
Email: {email}

Search Results:
{search_results}

Provide structured insights including:
- Company overview and recent developments
- Business opportunities for engagement
- Recommended engagement strategies
- Risk factors to consider
- Quality score for this contact (1-10)
- Specific next actions to take

Be concise and actionable in your recommendations.
For Search Results, focus on the most relevant and recent information and always include the Date and Source of the information.
"""

ACCOUNT_ANALYSIS_PROMPT_TEMPLATE = """
Analyze this account for comprehensive business intelligence:

Account: {account_name}
Additional Data: {account_data}

Market Research:
{search_results}

Provide analysis covering:
- Overall company health and stability
- Market position relative to competitors
- Growth potential score (1-10)
- Key competitors in their space
- Opportunities for account expansion
- Potential retention risks

Focus on actionable business intelligence.
For Market Research, always include the Date and Source of the information.
"""

SALES_SCRIPT_PROMPT_TEMPLATE = """
Create a personalized sales script for the following contact:

Contact: {contact_name}
Company: {company_name}
Title: {title}
Department: {department}

Script Requirements:
- Style: {script_style}
- Length: {script_length}

Contact AI Insights:
- Company Overview: {company_overview}
- Contact Quality Score: {contact_score}/10
- Business Opportunities: {business_opportunities}
- Engagement Strategy: {engagement_strategy}
- Risk Factors: {risk_factors}
- Recent News: {recent_news}

Generate a comprehensive sales script with all required components.
Make the script conversational, natural, and highly personalized.
"""
