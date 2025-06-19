import uvicorn
from fastapi import (
    Depends,
    Response,
    FastAPI,
    HTTPException,
    status,
    Form,
    Request,
    APIRouter,
)
from pydantic import BaseModel
import pandas as pd
from typing import Union, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-fitness-planner"

environment = os.getenv("ENVIRONMENT")
print("environment", environment)
import sys

if environment == "dev":
    print("detected dev environment")
    sys.path.append("fast_api")
else:
    sys.path.append("fast_api")

from app.api.nutrition_setup import nutrition_setup
from app.api.agents import agents
from app.api.nutrition_search import nutrition_search
from app.api.langgraph_agents import langgraph_agents


app = FastAPI(
    title="AI Fitness Planner API",
    version="1.0",
    description="== AI Fitness Planner API ==",
)


app.include_router(
    nutrition_setup,
    prefix="/v1/nutrition_setup",
    tags=["nutrition_setup"],
)

app.include_router(
    agents,
    prefix="/v1/agents",
    tags=["agents"],
)

app.include_router(
    nutrition_search,
    prefix="/v1/nutrition_search",
    tags=["nutrition_search"],
)

app.include_router(
    langgraph_agents,
    prefix="/v1/langgraph",
    tags=["langgraph_agents"],
)
