import os
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import httpx
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from datetime import datetime
import pandas as pd

# Load environment variables
load_dotenv()

web = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_dir = os.path.join(parent_dir, "static")

print("Static directory:", static_dir)

# Mount static files directory for images and other assets
web.mount("/static", StaticFiles(directory=static_dir), name="static")


# Database setup with fallback options
def get_working_database_uri():
    """Get a working database URI by testing connections"""
    # List of database URIs to try in order
    database_options = [
        "postgresql://postgres:Dev$1post23@bbdt_db:5432/postgres",  # From environment
        "postgresql://joshjanzen:normal@ai_clone_db:5432/ai_clone",  # From shiny_proxy pattern
        "postgresql://postgres:postgres@localhost:5432/postgres",  # Local fallback
    ]

    for uri in database_options:
        try:
            print(f"🔍 Testing database connection: {uri}")
            # Create engine with short timeout for testing
            test_engine = create_engine(uri, connect_args={"connect_timeout": 5})
            with test_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                print(f"✅ Database connection successful: {uri}")
                test_engine.dispose()  # Clean up test connection
                return uri
        except Exception as e:
            print(f"❌ Failed to connect to {uri}: {str(e)}")
            continue

    # If all fail, raise an error with details
    raise Exception(
        "❌ Could not connect to any database. Check your database configuration."
    )


# Get working database URI
try:
    DATABASE_URI = get_working_database_uri()
    print(f"🔍 Using DATABASE_URI: {DATABASE_URI}")
    engine = create_engine(DATABASE_URI)
except Exception as db_error:
    print(f"❌ Database setup failed: {db_error}")
    # Create a dummy engine that will fail gracefully
    DATABASE_URI = "postgresql://dummy:dummy@localhost:5432/dummy"
    engine = create_engine(DATABASE_URI)


# Pydantic model for unsubscribe request
class UnsubscribeRequest(BaseModel):
    email_address: str


# Pydantic model for chat request
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


def ensure_unsubscribe_table_exists():
    """Create the josh_janzen_unsubscribe table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS josh_janzen_unsubscribe (
        id SERIAL PRIMARY KEY,
        email_address VARCHAR(255) NOT NULL UNIQUE,
        unsubscribe_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        ip_address INET,
        user_agent TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create index for better query performance
    CREATE INDEX IF NOT EXISTS idx_josh_janzen_unsubscribe_email ON josh_janzen_unsubscribe(email_address);
    CREATE INDEX IF NOT EXISTS idx_josh_janzen_unsubscribe_timestamp ON josh_janzen_unsubscribe(unsubscribe_timestamp);
    """

    try:
        with engine.connect() as connection:
            connection.execute(text(create_table_sql))
            connection.commit()
            print("✅ Database table 'josh_janzen_unsubscribe' ensured to exist")
            return True
    except Exception as e:
        print(f"❌ Error creating unsubscribe table: {e}")
        return False


# Try to ensure table exists on startup (don't fail if it doesn't work)
try:
    table_created = ensure_unsubscribe_table_exists()
    if not table_created:
        print("⚠️ Warning: Could not create unsubscribe table at startup")
except Exception as e:
    print(f"⚠️ Warning: Failed to setup database table at startup: {e}")


@web.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path) as f:
        print("Serving index.html from:", index_path)
        return HTMLResponse(content=f.read())


@web.get("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page():
    unsubscribe_path = os.path.join(static_dir, "unsubscribe.html")
    with open(unsubscribe_path) as f:
        return HTMLResponse(content=f.read())


@web.get("/virtual_chat", response_class=HTMLResponse)
async def virtual_chat_page():
    virtual_chat_path = os.path.join(static_dir, "virtual_chat.html")
    with open(virtual_chat_path) as f:
        return HTMLResponse(content=f.read())


def save_unsubscribe_to_file(email_address: str, client_ip: str, user_agent: str):
    """Fallback: Save unsubscribe to CSV file when database is unavailable"""
    try:
        # Create unsubscribe data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), "unsubscribe_data")
        os.makedirs(data_dir, exist_ok=True)

        # CSV file path
        csv_file = os.path.join(data_dir, "josh_janzen_unsubscribe.csv")

        # Create data record
        unsubscribe_data = {
            "email_address": email_address,
            "unsubscribe_timestamp": datetime.now().isoformat(),
            "ip_address": client_ip,
            "user_agent": user_agent,
        }

        # Check if file exists and if email already unsubscribed
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            if email_address in existing_df["email_address"].values:
                print(f"ℹ️ Email already unsubscribed (file): {email_address}")
                return True, "Email already unsubscribed"

        # Append to CSV file
        df = pd.DataFrame([unsubscribe_data])
        df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)

        print(f"✅ Saved unsubscribe to file: {email_address}")
        return True, "Successfully unsubscribed"

    except Exception as e:
        print(f"❌ Error saving to file: {str(e)}")
        return False, f"Error saving unsubscribe: {str(e)}"


@web.post("/api/unsubscribe")
async def process_unsubscribe(request: UnsubscribeRequest, http_request: Request):
    """Process unsubscribe request and store in database or file fallback"""
    print(f"🔍 Received unsubscribe request for: {request.email_address}")

    # Get client IP and user agent
    client_ip = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("user-agent")

    print(f"🔍 Client IP: {client_ip}, User Agent: {user_agent}")

    # Try database first
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            print("✅ Database connection successful")

            # Create the data record
            unsubscribe_data = {
                "email_address": request.email_address,
                "unsubscribe_timestamp": datetime.now(),
                "ip_address": client_ip,
                "user_agent": user_agent,
            }

            # Convert to DataFrame and insert
            df = pd.DataFrame([unsubscribe_data])

            try:
                df.to_sql(
                    "josh_janzen_unsubscribe",
                    con=engine,
                    if_exists="append",
                    index=False,
                )
                print(f"✅ Processed unsubscribe (database): {request.email_address}")
                return JSONResponse(
                    content={"success": True, "message": "Successfully unsubscribed"}
                )

            except Exception as db_error:
                # Handle duplicate email addresses gracefully
                if (
                    "duplicate key" in str(db_error).lower()
                    or "unique constraint" in str(db_error).lower()
                ):
                    print(
                        f"ℹ️ Email already unsubscribed (database): {request.email_address}"
                    )
                    return JSONResponse(
                        content={
                            "success": True,
                            "message": "Email already unsubscribed",
                        }
                    )
                else:
                    raise db_error

    except Exception as db_error:
        print(f"❌ Database failed, using file fallback: {str(db_error)}")

        # Fallback to file storage
        success, message = save_unsubscribe_to_file(
            request.email_address, client_ip, user_agent
        )

        if success:
            return JSONResponse(content={"success": True, "message": message})
        else:
            return JSONResponse(
                content={"success": False, "message": message}, status_code=500
            )


@web.post("/virtual_chat_api")
async def virtual_chat_endpoint(request: ChatRequest) -> JSONResponse:
    """Chat endpoint for Josh Janzen virtual chat"""
    print(f"=== VIRTUAL CHAT REQUEST RECEIVED ===")
    print(f"Message: {request.message}")
    print(f"History length: {len(request.history)}")
    try:
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return JSONResponse(
                {"status": "error", "message": "OpenAI API key not configured"},
                status_code=500,
            )

        # Convert history to LangChain format
        chat_history = []
        for msg in request.history:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

        # Josh Janzen business context
        SYSTEM_MESSAGE = """You are a virtual AI assistant for Josh Janzen, a professional with expertise in technology, business, and AI systems.

ABOUT JOSH JANZEN:
- Technology professional with experience in AI, data science, and software development
- Business leader with expertise in strategic planning and operations
- Passionate about helping others learn and grow in technology
- Focus on practical applications of AI and automation in business

Contact
joshjanzen@gmail.com
www.linkedin.com/in/joshjanzen
(LinkedIn)
Top Skills
Generative AI
BEx Reporting
SAP ERP
Languages
English (Native or Bilingual)
Certifications
Machine Learning Certificate
Data Science Specialization
Large Language Models (LLMs),
Transformers & GPT A-Z
Python for Data Science
Honors-Awards
Pro Natural Physique Competitor
Josh Janzen
Data Science Leader
Minneapolis, Minnesota, United States
Summary
Data Science Director who transforms business vision into
measurable outcomes through AI and analytics initiatives. Proven
track record of delivering enterprise-wide data products that drive
significant value. Combines strategic leadership with deep technical
expertise to foster innovation and mentor next-generation talent.
Experience
C.H. Robinson
5 years 9 months
Director, Data Science
August 2021 - Present (3 years 11 months)
Greater Minneapolis-St. Paul Area
• Lead technical development leveraging advanced data science technical
stack including Azure, GenAI, Docker, Git, Python, and API development.
• Manage, mentor and coach Data Scientists, leading change on application
and data science initiatives.
Principal Data Scientist
October 2019 - August 2021 (1 year 11 months)
Greater Minneapolis-St. Paul Area
• Identified opportunities and built data science products involving 60% of
time building/implementing, 30% identifying possibilities, and 10% coaching/
developing team members.
• Dockerized data pipelines, creating RESTful API endpoints and various
integrations with other data services to deliver output via front-end and API.
Life Time Inc.
Senior Data Scientist
August 2018 - October 2019 (1 year 3 months)
Greater Minneapolis-St. Paul Area
• Using disparate data sources, created ETL solution to understand online
activity of members across mobile and web digital channels.
Page 1 of 3
Best Buy
8 years
Senior Data Scientist
2014 - 2018 (4 years)
United States
• Owned model development from ideation, data exploration, and testing.
• Processed, filtered, and aggregated large volumes of complex data from
various sources into actionable insights.
• Led, trained, and mentored an Associate Data Scientist.
Manager - Decision Support Analysis
2012 - 2014 (2 years)
Deep-dive investigation leveraging various data sources to uncover business
opportunities and drive outcomes.
Decision support financial analysis for Product Acquisition and Flow capability.
Role and team form critical part of the 3yr $800M cost savings initiative.
Created an Open Box pricing model and business lead for IT enhancement
which delivers systematic Open Box pricing to Stores. Projected gross profit
value annualized at $7.4M. Project led to nomination for BTO (Business Team
Operations) August 2012 Recognition award.
Identify process gaps across major product returns expense categories,
including Flat-Panel TVs of over $10M identified as an issue.
Senior Financial Analyst
2010 - 2012 (2 years)
Modeled and established process to improve Owned Inventory by Class;
results are 23% better than goal. Trained other CSGs the methodology,
process, and tool. Project contributed to Best Buy 2011 Finance Team World
Class award.
Led a cross-functional project and made recommendations on future retail
space 3yr financial plan. Partnered with Space and Operations to identify
constraints and costs to repurpose space to improve Gross Profit Return on
Space (GPROS) by 9% from FY12-FY15.
Created a driver-based P&L model of a new business partnership for Mobile
Electronics to drive potential Revenue and Operating Profit over FY13-FY17.
Page 2 of 3
Model provided ability to adjust market ramp and P&L scenarios to assist
leadership in partnership negotiations.
Chico's FAS Inc.
Senior Planning Analyst
2009 - 2010 (1 year)
Fort Myers, Florida Area
• SAP implementation, serving as BI Lead, for merchandise planning leading to
seamless reporting and on-time implementation launch.
Skyline Exhibits
Senior Financial Analyst
2006 - 2009 (3 years)
• Support global initiatives in SAP Business Intelligence & Business
Warehouse, along with BEx reporting - SAP Super-user.
Target
Senior Business Analyst
2000 - 2004 (4 years)
• Consulted specific departments for Demand Planning, driving company
initiatives on top-selling, high margin product.
• Promoted 2 times due to proven expertise, results, and people leadership
skills.
Education
University of Minnesota
BS, Applied Economics
University of St. Thomas
Master of Science (M.S.), Data Science
University of St. Thomas - Opus College of Business
Master's, M.B.A
Page 3 of 3

COMMUNICATION STYLE:
- Professional yet approachable
- Clear and concise explanations
- Focus on practical, actionable advice
- Helpful and supportive tone

TOPICS YOU CAN HELP WITH:
- Technology and AI questions
- Business strategy and operations
- Career advice in tech
- Learning programming and data science
- AI tools and automation
- General professional development

Keep responses helpful, professional, and focused on providing value. If asked about topics outside your expertise, not related to AI/Data Science, or inappropriate content, politely redirect to areas where you can be most helpful.

Answer questions naturally as if you were Josh Janzen's virtual assistant, representing his knowledge and approach to helping others.
"""

        # Initialize ChatOpenAI with explicit parameters and error handling
        try:
            llm = ChatOpenAI(temperature=0.7, model="gpt-4o", openai_api_key=api_key)
        except TypeError as e:
            if "proxies" in str(e):
                # Fallback for version compatibility issues
                os.environ["OPENAI_API_KEY"] = api_key
                llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
            else:
                raise e

        # Create messages with LangChain format
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            *chat_history,
            HumanMessage(content=request.message),
        ]

        print(f"Virtual chat request: {request.message}")
        print(f"Total messages: {len(messages)}")

        # Get response from LangChain
        response = llm.invoke(messages)

        return JSONResponse({"response": response.content, "status": "success"})

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Error in virtual chat endpoint: {str(e)}")
        print(f"Full traceback: {error_details}")
        print(
            f"OpenAI API Key present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}"
        )
        return JSONResponse(
            {
                "status": "error",
                "message": "Sorry, I'm having trouble right now. Please try again later.",
                "debug": str(e) if os.getenv("DEBUG") else None,
            },
            status_code=500,
        )
