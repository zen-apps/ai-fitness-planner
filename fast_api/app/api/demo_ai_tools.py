from fastapi import APIRouter, Response, Request
import json
import logging
import requests
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import MessagesPlaceholder
from pydantic import BaseModel
import random
from langchain_community.utilities import SerpAPIWrapper

logger = logging.getLogger(__name__)
demo_ai_tools = APIRouter()

# ----------------------------- josh FLIGHT AI ASSISTANT ----------------------------- #
ticket_prices = {
    "london": "$799",
    "paris": "$899",
    "tokyo": "$1400",
    "berlin": "$499",
    "rome": "$749",
    "barcelona": "$679",
    "sydney": "$1650",
    "singapore": "$1200",
    "dubai": "$950",
    "new york": "$650",
    "los angeles": "$550",
    "chicago": "$450",
    "miami": "$520",
    "bangkok": "$890",
    "cairo": "$780",
    "amsterdam": "$590",
    "istanbul": "$670",
    "phoenix": "$480",
    "minneapolis": "$530",
}

# Flight status database (mock data)
flight_statuses = {
    "FA100": {
        "status": "On Time",
        "departure": "10:30 AM",
        "arrival": "1:45 PM",
        "gate": "B12",
    },
    "FA202": {
        "status": "Delayed",
        "departure": "2:15 PM",
        "arrival": "5:00 PM",
        "gate": "C4",
    },
    "FA305": {
        "status": "Boarding",
        "departure": "3:30 PM",
        "arrival": "6:45 PM",
        "gate": "A7",
    },
    "FA410": {
        "status": "Cancelled",
        "departure": "7:00 AM",
        "arrival": "9:30 AM",
        "gate": "D2",
    },
    "FA520": {
        "status": "Landed",
        "departure": "8:45 AM",
        "arrival": "11:15 AM",
        "gate": "E9",
    },
}

# Airport information database
airport_info = {
    "lhr": {
        "name": "London Heathrow",
        "city": "London",
        "country": "United Kingdom",
        "terminals": 5,
    },
    "cdg": {
        "name": "Charles de Gaulle",
        "city": "Paris",
        "country": "France",
        "terminals": 3,
    },
    "hnd": {
        "name": "Tokyo Haneda",
        "city": "Tokyo",
        "country": "Japan",
        "terminals": 3,
    },
    "txl": {
        "name": "Berlin Tegel",
        "city": "Berlin",
        "country": "Germany",
        "terminals": 1,
    },
    "fco": {
        "name": "Rome Fiumicino",
        "city": "Rome",
        "country": "Italy",
        "terminals": 4,
    },
    "bcn": {
        "name": "Barcelona El Prat",
        "city": "Barcelona",
        "country": "Spain",
        "terminals": 2,
    },
    "syd": {
        "name": "Sydney International",
        "city": "Sydney",
        "country": "Australia",
        "terminals": 3,
    },
    "jfk": {
        "name": "John F. Kennedy",
        "city": "New York",
        "country": "USA",
        "terminals": 6,
    },
    "lax": {
        "name": "Los Angeles International",
        "city": "Los Angeles",
        "country": "USA",
        "terminals": 9,
    },
}

# Baggage policies
baggage_policies = {
    "economy": {
        "carry_on": "1 bag up to 7kg",
        "checked": "1 bag up to 23kg",
        "fee_extra_bag": "$50",
    },
    "premium_economy": {
        "carry_on": "1 bag up to 7kg",
        "checked": "2 bags up to 23kg each",
        "fee_extra_bag": "$40",
    },
    "business": {
        "carry_on": "2 bags up to 7kg each",
        "checked": "2 bags up to 32kg each",
        "fee_extra_bag": "$0",
    },
    "first": {
        "carry_on": "2 bags up to 7kg each",
        "checked": "3 bags up to 32kg each",
        "fee_extra_bag": "$0",
    },
}

# Travel requirements database
travel_requirements = {
    "united kingdom": {
        "passport": "Required",
        "visa": "Required for some countries",
        "covid": "No restrictions",
    },
    "france": {
        "passport": "Required",
        "visa": "Not required for EU citizens",
        "covid": "No restrictions",
    },
    "japan": {
        "passport": "Required",
        "visa": "Required for most countries",
        "covid": "Proof of vaccination recommended",
    },
    "germany": {
        "passport": "Required",
        "visa": "Not required for EU citizens",
        "covid": "No restrictions",
    },
    "australia": {
        "passport": "Required",
        "visa": "Required for most countries",
        "covid": "No restrictions",
    },
    "united states": {
        "passport": "Required",
        "visa": "Required for most countries",
        "covid": "No restrictions",
    },
    "thailand": {
        "passport": "Required",
        "visa": "Required for stays over 30 days",
        "covid": "No restrictions",
    },
}

# -------------------- TOOL FUNCTIONS -------------------- #


def get_ticket_price(destination_city: str) -> str:
    """Get the price of a return ticket to the destination city."""
    city = destination_city.lower()
    return ticket_prices.get(
        city, "Price information not available for this destination"
    )


def get_flight_status(flight_number: str) -> str:
    """Get the status of a flight by flight number."""
    flight = flight_number.upper()
    if flight in flight_statuses:
        status = flight_statuses[flight]
        return f"Flight {flight}: {status['status']}. Departure: {status['departure']}, Arrival: {status['arrival']}, Gate: {status['gate']}"
    return "Flight information not found"


def get_airport_info(airport_code: str) -> str:
    """Get information about an airport by its IATA code."""
    code = airport_code.lower()
    if code in airport_info:
        info = airport_info[code]
        return f"{info['name']} ({code.upper()}) is located in {info['city']}, {info['country']} and has {info['terminals']} terminal(s)."
    return "Airport information not found"


def get_baggage_policy(fare_class: str) -> str:
    """Get baggage allowance information for a specific fare class."""
    fare = fare_class.lower()
    if fare in baggage_policies:
        policy = baggage_policies[fare]
        return f"Baggage allowance for {fare.title()} class: Carry-on: {policy['carry_on']}, Checked: {policy['checked']}, Extra bag fee: {policy['fee_extra_bag']}"
    return "Baggage policy not found for this fare class"


def get_travel_requirements(country: str) -> str:
    """Get travel requirements for a specific country."""
    country_lower = country.lower()
    if country_lower in travel_requirements:
        req = travel_requirements[country_lower]
        return f"Travel requirements for {country.title()}: Passport: {req['passport']}, Visa: {req['visa']}, COVID-19: {req['covid']}"
    return "Travel requirement information not available for this country"


def check_available_seats(flight_date: str, destination: str) -> str:
    """Check if seats are available for a flight on a specific date to a destination."""
    # This is a mock function that would normally query a real database
    try:
        # Parse the date string
        date_parts = flight_date.split("-")
        if len(date_parts) == 3:
            year, month, day = (
                int(date_parts[0]),
                int(date_parts[1]),
                int(date_parts[2]),
            )
            # Generate a random number of seats (1-50) based on the date and destination
            seats = random.randint(1, 50)
            return f"There are approximately {seats} seats available for flights to {destination} on {flight_date}."
        return "Please provide the date in YYYY-MM-DD format"
    except Exception as e:
        return f"Error checking seat availability: {str(e)}"


def get_weather_forecast(city: str) -> str:
    """Get the weather forecast for a city (mock data)."""
    # This would typically call a weather API
    weather_conditions = [
        "Sunny",
        "Partly Cloudy",
        "Cloudy",
        "Rainy",
        "Thunderstorms",
        "Snowy",
        "Windy",
    ]
    temp_range = (15, 30)  # Celsius

    # Deterministic "random" selection based on city name
    city_hash = sum(ord(c) for c in city.lower())
    weather_index = city_hash % len(weather_conditions)
    temp = city_hash % (temp_range[1] - temp_range[0]) + temp_range[0]

    return f"The weather forecast for {city.title()} is {weather_conditions[weather_index]} with temperatures around {temp}°C."


def recommend_destination(budget: str, preferences: str) -> str:
    """Recommend a destination based on budget and preferences."""
    budget_value = 0
    try:
        # Extract numeric value from budget string
        import re

        budget_match = re.search(r"\d+", budget)
        if budget_match:
            budget_value = int(budget_match.group())
    except:
        budget_value = 800  # Default budget

    preferences_lower = preferences.lower()

    # Logic to recommend destinations based on budget and preferences
    if "beach" in preferences_lower:
        if budget_value < 600:
            return "Given your budget and beach preference, consider Miami or Cancun for beautiful beaches without breaking the bank."
        else:
            return "With your budget and beach preference, consider Bali, Thailand, or the Maldives for stunning tropical experiences."
    elif "culture" in preferences_lower or "history" in preferences_lower:
        if budget_value < 600:
            return "For cultural experiences within your budget, consider Prague, Budapest, or Mexico City - all rich in history and architecture."
        else:
            return "With your budget and interest in culture, Rome, Athens, or Kyoto would offer incredible historical and cultural experiences."
    elif "nature" in preferences_lower or "outdoor" in preferences_lower:
        if budget_value < 600:
            return "For nature and outdoor activities within your budget, consider Costa Rica, Colorado, or the Canadian Rockies."
        else:
            return "With your budget and love for nature, New Zealand, Iceland, or Switzerland would offer breathtaking landscapes and outdoor adventures."
    else:
        if budget_value < 600:
            return "For a budget-friendly trip around $500-600, consider Berlin, Lisbon, or Bangkok for great value and diverse experiences."
        else:
            return "With a comfortable budget of $800+, Tokyo, London, or Sydney would offer excellent experiences with diverse attractions."


def miles_calculator(miles: int) -> str:
    """Calculate what rewards can be redeemed with a given number of miles."""
    if miles < 10000:
        return (
            f"With {miles} miles, you can redeem a seat upgrade on a domestic flight."
        )
    elif miles < 25000:
        return f"With {miles} miles, you can redeem a free domestic round-trip ticket in economy class."
    elif miles < 50000:
        return f"With {miles} miles, you can redeem a free international round-trip ticket to Europe or Latin America in economy class."
    elif miles < 100000:
        return f"With {miles} miles, you can redeem a free international round-trip ticket to Europe in business class or to Asia in economy class."
    else:
        return f"With {miles} miles, you can redeem a free international round-trip ticket to anywhere in the world in business class, or to select destinations in first class."


# -------------------- API ENDPOINTS -------------------- #


class SearchRequest(BaseModel):
    query: str


@demo_ai_tools.post("/search")
async def search_endpoint(request: SearchRequest):
    """Endpoint to handle search requests using SerpAPI."""
    try:
        search = SerpAPIWrapper()
        results = search.run(request.query)
        return {"results": results, "status": "success"}
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}


@demo_ai_tools.post("/chat")
async def chat_endpoint(query: Dict[Any, Any]) -> Response:
    """Chat endpoint that handles the conversation with the AI."""
    try:
        # Extract the message and history from the query
        message = query.get("message", "")
        history = query.get("history", [])

        # Convert history to the format LangChain expects
        chat_history = []
        for msg in history:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

        # System message
        SYSTEM_MESSAGE = """You are a helpful assistant for an Airline called FlightAI. 
        Give courteous and informative answers, no more than 3 sentences. 
        Always be accurate and use the appropriate tools when available.
        If you don't know the answer, say so and make a suggestion.
        When asked about flight prices, flight status, airport information, baggage policies, or travel requirements,
        make sure to use the appropriate tool to get the most up-to-date information."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_MESSAGE),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Set up all the tools
        tools = [
            Tool(
                name="get_ticket_price",
                func=get_ticket_price,
                description="Get the price of a return ticket to the destination city. Input should be a city name like 'London' or 'Tokyo'.",
            ),
            Tool(
                name="get_flight_status",
                func=get_flight_status,
                description="Get the status of a flight by its flight number. Input should be a flight number like 'FA100' or 'FA202'.",
            ),
            Tool(
                name="get_airport_info",
                func=get_airport_info,
                description="Get information about an airport by its IATA code. Input should be an airport code like 'LHR' or 'JFK'.",
            ),
            Tool(
                name="get_baggage_policy",
                func=get_baggage_policy,
                description="Get baggage allowance information for a specific fare class. Input should be a fare class like 'economy', 'business', or 'first'.",
            ),
            Tool(
                name="get_travel_requirements",
                func=get_travel_requirements,
                description="Get travel requirements for a specific country. Input should be a country name like 'United Kingdom' or 'Japan'.",
            ),
            Tool(
                name="check_available_seats",
                func=check_available_seats,
                description="Check if seats are available for a flight on a specific date to a destination. Input should be in format 'YYYY-MM-DD, Destination' like '2023-12-25, London'.",
            ),
            Tool(
                name="get_weather_forecast",
                func=get_weather_forecast,
                description="Get the weather forecast for a city. Input should be a city name like 'London' or 'Tokyo'.",
            ),
            Tool(
                name="recommend_destination",
                func=recommend_destination,
                description="Recommend a destination based on budget and preferences. Input should be in format 'Budget: $X, Preferences: Y' like 'Budget: $800, Preferences: beaches and relaxation'.",
            ),
            Tool(
                name="miles_calculator",
                func=miles_calculator,
                description="Calculate what rewards can be redeemed with a given number of miles. Input should be a number like '25000'.",
            ),
        ]

        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        llm_with_tools = llm.bind(
            functions=[convert_to_openai_function(t) for t in tools]
        )

        # Create the agent
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        result = agent_executor.invoke({"input": message, "chat_history": chat_history})
        response = {"response": result["output"], "status": "success"}

        return Response(json.dumps(response), media_type="application/json")

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        error_response = {"status": "error", "message": str(e)}
        return Response(
            json.dumps(error_response), media_type="application/json", status_code=500
        )


# ----------------------------- josh HELLO WORLD TO EACH MODEL ----------------------------- #
@demo_ai_tools.post("/hello_gemini")
async def hello_gemini(query: Dict[Any, Any]) -> Response:
    """{"question": "test"}"""
    try:
        question = query["question"]
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # or "gemini-pro-vision" if you need image capabilities
            temperature=0,
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question),
        ]
        response = llm(messages)
        response = {"message": response.content}
        return Response(json.dumps(response), media_type="application/json")
    except Exception as e:
        logger.error(f"Error in hello_gemini endpoint: {str(e)}")
        return Response(
            json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


@demo_ai_tools.post("/hello_openai")
async def hello_openai(query: Dict[Any, Any]) -> Response:
    """{"question": "test"}"""
    try:
        question = query["question"]
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question),
        ]
        response = llm(messages)
        response = {"message": response.content}
        return Response(json.dumps(response), media_type="application/json")
    except Exception as e:
        logger.error(f"Error in hello_openai endpoint: {str(e)}")
        return Response(
            json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


@demo_ai_tools.post("/hello_anthropic")
async def hello_anthropic(query: Dict[Any, Any]) -> Response:
    """{"question": "test"}"""
    try:
        question = query["question"]
        model = ChatAnthropic(model="claude-3-5-sonnet-latest")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question),
        ]
        response = model(messages)
        response = {"message": response.content}
        return Response(json.dumps(response), media_type="application/json")
    except Exception as e:
        logger.error(f"Error in hello_anthropic endpoint: {str(e)}")
        return Response(
            json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


@demo_ai_tools.post("/hello_local_ollama")
async def hello_local_ollama(query: dict) -> Response:
    """
    Connects to local Ollama endpoint.
    Example input: {"question": "test"}
    """
    try:
        question = query["question"]

        url = "http://107.115.203.58:1001"
        endpoint = "/v1/genai/connect_local_ollama/"
        api_url = url + endpoint

        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # Send as form data to match the receiving endpoint's expectations
        data = {"question": question}

        response = requests.post(api_url, headers=headers, data=data)
        return Response(response.text, media_type="application/json")

    except Exception as e:
        logger.error(f"Error in hello_local_ollama: {str(e)}")
        return Response(
            json.dumps({"response": None, "error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


# -------------------------- josh audio extraction and recap

from fastapi import APIRouter, Response, Request, UploadFile, File, Form
import json
import logging
import tempfile
import os
from typing import Dict, Any
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@demo_ai_tools.post("/transcribe-audio")
async def transcribe_audio(
    audio: UploadFile = File(...),
    instructions: str = Form(
        default="Please provide a detailed summary of the audio content."
    ),
):
    """
    Endpoint to transcribe audio using OpenAI Whisper and generate a summary using GPT-4.

    Args:
        audio (UploadFile): The audio file to transcribe
        instructions (str): Specific instructions for summarizing the audio
    """
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(audio.filename)[1]
        ) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Transcribe using Whisper
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

        print("***************", transcription, instructions)
        # Clean up temporary file
        os.unlink(temp_file_path)

        # Initialize ChatGPT for summarization
        llm = ChatOpenAI(temperature=0, model="gpt-4o")

        # Create prompt for summarization
        messages = [
            SystemMessage(
                content="You are an AI assistant that generates concise and accurate summaries of audio transcriptions."
            ),
            HumanMessage(
                content=f"""
            Please analyze this transcription and provide a summary according to these instructions: {instructions}
            
            Transcription:
            {transcription}
            """
            ),
        ]

        # Get summary
        summary_response = llm.invoke(messages)
        print("summary res", summary_response)
        summary = summary_response.content

        response = {
            "status": "success",
            "transcription": transcription,
            "summary": summary,
        }

        return Response(json.dumps(response), media_type="application/json")

    except Exception as e:
        logger.error(f"Error in transcribe_audio endpoint: {str(e)}")
        error_response = {"status": "error", "message": str(e)}
        return Response(
            json.dumps(error_response), media_type="application/json", status_code=500
        )


@demo_ai_tools.post("/audio_chat")
async def audio_chat(query: Dict[Any, Any]) -> Response:
    """
    Chat endpoint for discussing audio transcripts. Maintains context about the transcript
    and allows for follow-up questions.

    Expected input format:
    {
        "message": str,            # The user's question/message
        "transcript": str,         # The audio transcript to discuss
        "chat_history": List[Dict] # Previous chat messages
    }
    """
    try:
        # Extract information from the query
        message = query.get("message", "")
        transcript = query.get("transcript", "")
        history = query.get("chat_history", [])

        # Convert history to LangChain format
        chat_history = []
        for msg in history:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

        # Initialize ChatGPT
        llm = ChatOpenAI(temperature=0, model="gpt-4o")

        # Create the system message with context about the transcript
        system_message = SystemMessage(
            content="""You are a helpful AI assistant analyzing an audio transcript. 
        Your role is to help users understand and analyze the content of the transcript.
        Provide specific references to the transcript when relevant.
        If asked about something not in the transcript, make it clear that the information isn't available in the recording."""
        )

        # Create the context message containing the transcript
        context_message = SystemMessage(
            content=f"""Here is the transcript of the audio recording:
        {transcript}
        
        Please answer questions about this transcript, providing specific details when possible."""
        )

        # Combine all messages
        messages = [
            system_message,
            context_message,
            *chat_history,
            HumanMessage(content=message),
        ]

        # Get response from LLM
        response = llm.invoke(messages)

        # Prepare the response
        chat_response = {"response": response.content, "status": "success"}

        return Response(json.dumps(chat_response), media_type="application/json")

    except Exception as e:
        logger.error(f"Error in audio_chat endpoint: {str(e)}")
        error_response = {"status": "error", "message": str(e)}
        return Response(
            json.dumps(error_response), media_type="application/json", status_code=500
        )


# ----------------------------------- josh ai customer router
import os
import json
import logging
from fastapi import APIRouter, Response
from typing import List, TypedDict, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

import pandas as pd


# Pydantic models remain unchanged...
class AnalyzePriority(BaseModel):
    priority: str = Field(
        description="The priority level of the support ticket (High/Medium/Low)"
    )


class ClassifyDepartment(BaseModel):
    department: str = Field(
        description="The department that should handle the ticket (Technical/Billing/Sales)"
    )


class MakeModelYear(BaseModel):
    make: str = Field(description="The make of the car")
    model: str = Field(description="The model of the car")
    year: str = Field(description="The year of the car")


class TicketState(TypedDict):
    ticket_text: str
    priority: str
    department: str
    response: str
    inventory_data: List[Dict[str, Any]]


class InventoryQuery(BaseModel):
    make: str = Field(description="The make of the car")
    model: Optional[str] = Field(description="The model of the car")
    year: Optional[str] = Field(description="The year of the car")


class InventoryResult(BaseModel):
    items: List[Dict[str, Any]] = Field(description="List of matching inventory items")
    count: int = Field(description="Number of matching items found")
    message: str = Field(description="Summary message about the results")


def get_car_part_inventory() -> List[Dict[str, Any]]:
    """Get the current inventory of car parts."""
    logger.info("Fetching car part inventory")
    file_name = "inventory_data.csv"
    full_path = f"./fast_api_ai_clone/app/api/data/{file_name}"
    df = pd.read_csv(full_path)
    car_part_inventory = df.to_dict(orient="dict")
    print(df.head())
    logger.info(f"Retrieved inventory with {len(car_part_inventory['make'])} items")
    return car_part_inventory


@tool
def search_inventory(query: InventoryQuery) -> InventoryResult:
    """
    Search the car parts inventory based on make, model, and year.
    If model or year are not provided, returns all matches for the make.

    Args:
        query (InventoryQuery): Search criteria including make, model, and year

    Returns:
        InventoryResult: Matching inventory items and summary
    """
    logger.info(f"Searching inventory for: {query.make} {query.model} {query.year}")

    inventory_data = get_car_part_inventory()
    filtered_inventory = []

    # Convert search terms to lowercase for case-insensitive matching
    search_make = query.make.lower()
    search_model = query.model.lower() if query.model is not None else ""
    search_year = query.year if query.year is not None else ""

    print("*********", search_make, search_model, search_year)

    for idx in range(len(inventory_data["make"])):
        current_make = inventory_data["make"][idx].lower()
        current_model = inventory_data["model"][idx].lower()
        current_year = inventory_data["year"][idx]

        # Check if item matches search criteria
        make_matches = search_make in current_make
        model_matches = not search_model or search_model in current_model
        year_matches = not search_year or str(search_year) == str(current_year)

        if make_matches and model_matches and year_matches:
            filtered_inventory.append(
                {
                    "make": inventory_data["make"][idx],
                    "model": inventory_data["model"][idx],
                    "year": inventory_data["year"][idx],
                    "part_id": inventory_data["part_id"][idx],
                    "part": inventory_data["part"][idx],
                    "instock": inventory_data["instock"][idx],
                    "pricing": inventory_data["pricing"][idx],
                    "shipping": inventory_data["shipping"][idx],
                }
            )

    # Create result message
    if filtered_inventory:
        message = f"Found {len(filtered_inventory)} matching items for {query.make}"
        if query.model:
            message += f" {query.model}"
        if query.year:
            message += f" ({query.year})"
    else:
        message = f"No inventory items found matching {query.make}"
        if query.model:
            message += f" {query.model}"
        if query.year:
            message += f" ({query.year})"

    return InventoryResult(
        items=filtered_inventory, count=len(filtered_inventory), message=message
    )


def check_inventory(state: TicketState) -> TicketState:
    """Check inventory if department is Sales using the inventory search tool"""
    logger.info(f"Starting inventory check for department: {state['department']}")

    if state["department"] == "Sales":
        OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(api_key=OPENAPI_KEY, model="gpt-4o", temperature=0.0)

        messages = [
            SystemMessage(
                content="Extract the car make, model, and year if present. Return structured data with these fields."
            ),
            HumanMessage(
                content=f"Extract car information from this ticket:\n\n{state['ticket_text']}"
            ),
        ]

        structured_llm = llm.with_structured_output(InventoryQuery)
        car_info = structured_llm.invoke(messages)

        # Create the query with the required "query" key
        query_input = {
            "query": {
                "make": car_info.make,
                "model": car_info.model,
                "year": car_info.year,
            }
        }

        print("*********", query_input)

        # Use the inventory search tool
        result = search_inventory(query_input)
        logger.info(result.message)

        state["inventory_data"] = result.items
    else:
        logger.info("Skipping inventory check for non-Sales department")
        state["inventory_data"] = []

    return state


def analyze_priority(state: TicketState) -> TicketState:
    """Determine ticket priority level based on content analysis"""
    logger.info("Starting priority analysis")
    logger.debug(f"Analyzing ticket text: {state['ticket_text'][:100]}...")

    OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAPI_KEY, model="gpt-4o", temperature=0.0)

    messages = [
        SystemMessage(content="""You are a support ticket analyzer..."""),
        HumanMessage(content=f"Analyze this support ticket:\n\n{state['ticket_text']}"),
    ]

    structured_llm = llm.with_structured_output(AnalyzePriority)
    response = structured_llm.invoke(messages)
    state["priority"] = response.priority

    logger.info(f"Priority analysis complete. Result: {response.priority}")
    return state


def classify_department(state: TicketState) -> TicketState:
    """Determine which department should handle the ticket"""
    logger.info("Starting department classification")
    logger.debug(f"Current ticket priority: {state['priority']}")

    OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAPI_KEY, model="gpt-4o", temperature=0.0)

    messages = [
        SystemMessage(content="You are a support ticket classifier..."),
        HumanMessage(
            content=f"Classify this support ticket:\n\n{state['ticket_text']}"
        ),
    ]

    structured_llm = llm.with_structured_output(ClassifyDepartment)
    response = structured_llm.invoke(messages)
    state["department"] = response.department

    logger.info(f"Department classification complete. Result: {response.department}")
    return state


def generate_response(state: TicketState) -> TicketState:
    """Generate appropriate response based on department, priority, and inventory"""
    logger.info("Starting response generation")
    logger.debug(
        f"Current state - Priority: {state['priority']}, Department: {state['department']}, Inventory items: {len(state['inventory_data'])}"
    )

    OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAPI_KEY, model="gpt-4o", temperature=0.0)

    inventory_context = ""
    if state["department"] == "Sales" and state["inventory_data"]:
        inventory_items = "\n".join(
            [
                f"- {item['part']} for {item['make']} {item['model']} ({item['year']}): "
                f"${item['pricing']} + ${item['shipping']} shipping "
                f"({'In Stock' if item['instock'] else 'Out of Stock'})"
                for item in state["inventory_data"]
            ]
        )
        inventory_context = f"\n\nAvailable inventory:\n{inventory_items}"
        logger.debug(f"Added inventory context: {inventory_context}")

    messages = [
        SystemMessage(
            content=f"""You are a {state['department'].lower()} support specialist
            RULES:
            1. For Inventory, use a general match based on Make and Model.  Do not worry about specific size.
            2. DO NOT MAKE UP ANY PRODUCTS THAT ARE NOT IN THE INVENTORY."""
        ),
        HumanMessage(
            content=f"Generate a response for this ticket:\n\n{state['ticket_text']}"
        ),
    ]

    response = llm.invoke(messages)
    state["response"] = response.content
    logger.info("Response generation complete")
    logger.debug(f"Generated response length: {len(response.content)}")

    return state


def create_ticket_workflow() -> StateGraph:
    """Create and configure the workflow with proper state handling."""
    logger.info("Creating ticket workflow")
    workflow = StateGraph(TicketState)

    # Add nodes
    workflow.add_node("analyze_priority", analyze_priority)
    workflow.add_node("classify_department", classify_department)
    workflow.add_node("check_inventory", check_inventory)
    workflow.add_node("generate_response", generate_response)

    # Define edges
    workflow.add_edge("analyze_priority", "classify_department")
    logger.debug("Added workflow edges")

    def should_check_inventory(state: TicketState) -> bool:
        should_check = state["department"] == "Sales"
        logger.debug(f"Checking if inventory check needed: {should_check}")
        return should_check

    workflow.add_conditional_edges(
        "classify_department",
        should_check_inventory,
        {True: "check_inventory", False: "generate_response"},
    )

    workflow.add_edge("check_inventory", "generate_response")
    workflow.add_edge("generate_response", END)

    workflow.set_entry_point("analyze_priority")
    logger.info("Workflow creation complete")

    return workflow.compile()


@demo_ai_tools.post("/process_support_ticket/")
async def process_support_ticket(query: dict) -> Response:
    """Process a support ticket through the workflow."""
    logger.info("=== Starting new support ticket workflow ===")
    logger.debug(f"Received query: {query}")

    ticket_text = query.get("ticket_text", "")
    logger.info(f"Processing ticket (first 100 chars): {ticket_text[:100]}...")

    # Initialize workflow
    app = create_ticket_workflow()

    # Initialize state
    initial_state: TicketState = {
        "ticket_text": ticket_text,
        "priority": "",
        "department": "",
        "response": "",
        "inventory_data": [],
    }

    logger.info("Executing workflow")
    # Run workflow
    result = app.invoke(initial_state)

    output_dict = {
        "workflow_response": {
            "priority": result.get("priority", ""),
            "department": result.get("department", ""),
            "response": result.get("response", ""),
            "inventory_data": result.get("inventory_data", []),
        }
    }

    logger.info("Workflow execution complete")
    logger.debug(
        f"Final result - Priority: {result.get('priority')}, Department: {result.get('department')}"
    )
    logger.info("=== Support ticket workflow finished ===")

    return Response(json.dumps(output_dict), media_type="application/json")
