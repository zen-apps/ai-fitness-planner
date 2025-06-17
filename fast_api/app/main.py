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

environment = os.getenv("ENVIRONMENT")
print("environment", environment)
import sys

if environment == "dev":
    print("detected dev environment")
    sys.path.append("fast_api")
else:
    sys.path.append("fast_api")

from app.api.workout import workout
from app.api.web import web

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AI Fitness Planner API",
    version="1.0",
    description="== AI Fitness Planner API ==",
)

# Mount static files at root level
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

login = APIRouter()


app.include_router(
    workout,
    prefix="/v1/workout",
    tags=["workout"],
)

app.include_router(
    web,
    prefix="/v1/web",
    tags=["web"],
)

# Also include web routes at root level for public pages
app.include_router(
    web,
    prefix="",
    tags=["web-public"],
)
