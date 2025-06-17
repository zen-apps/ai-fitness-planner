from fastapi import APIRouter, Response, Request
import json
import logging
from datetime import datetime
import pandas as pd
import requests
import os

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

genai = APIRouter()


# @genai.post("/tools_structured_single_output_parser")
