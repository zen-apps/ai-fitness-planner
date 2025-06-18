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
from app.api.nutrition import nutrition


app = FastAPI(
    title="AI Fitness Planner API",
    version="1.0",
    description="== AI Fitness Planner API ==",
)


app.include_router(
    workout,
    prefix="/v1/workout",
    tags=["workout"],
)

app.include_router(
    nutrition,
    prefix="/v1/nutrition",
    tags=["nutrition"],
)
