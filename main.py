import os, datetime, requests
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from serpapi import GoogleSearch

# Security imports
from security.output.safe_parser import SafeParser
from security.input.validator import (
    InputValidator,
    ExtractedTravelInfo,
    RefinedTravelInfo
)
from security.prompt.hardened_templates import HardenedPromptTemplates
from security.output.pii_detector import PIIDetector
from security.logging.security_logger import SecurityLogger, SecurityEventType
from security.middleware import SecurityMiddleware

# ENV
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "travel-planner-ai"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# Core
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()
memory = ConversationBufferMemory(return_messages=True)
weather_util = OpenWeatherMapAPIWrapper()
search_util = SerpAPIWrapper()

# Security components
security_logger = SecurityLogger()
safe_parser = SafeParser(logger=security_logger)
pii_detector = PIIDetector(logger=security_logger)
security_middleware = SecurityMiddleware(logger=security_logger)

# State
class TravelPlannerState(TypedDict, total=False):
    user_input: str
    session_id: str
    security_passed: bool
    security_block_reason: Optional[str]
    departure_city: str
    destination_country: str
    preferences: str
    budget: str
    outbound_date: str
    return_date: str
    destination_city: str
    arrival_airport: str
    flight: Dict[str, Dict[str, Any]]
    hotel: Dict[str, Dict[str, Any]]
    total: str
    weather: str
    places: str
    itinerary: str
    notes: str

# -------------------- TOOLS --------------------

@tool
def estimate_flight_cost(departure_city: str, destination_city: str, outbound_date: str, return_date: str) -> dict:
    """Use SerpAPI to get detailed flight listings (nested dict)."""
    api_key = os.getenv("SERPAPI_API_KEY")
    iata_map = {
        "New York": "JFK", "Paris": "CDG", "London": "LHR",
        "Tokyo": "HND", "Los Angeles": "LAX", "Dubai": "DXB", "Lahore": "LHE"
    }
    dep_code = iata_map.get(departure_city, departure_city)
    dest_code = iata_map.get(destination_city, destination_city)
    params = {
        "engine": "google_flights",
        "departure_id": dep_code,
        "arrival_id": dest_code,
        "outbound_date": outbound_date,
        "return_date": return_date,
        "currency": "USD",
        "hl": "en",
        "api_key": api_key
    }

    try:
        flight_results = GoogleSearch(params).get_dict()
        flights = flight_results.get("best_flights", [])
        all_flights_data = {}

        for idx, flight in enumerate(flights[:5], start=1):
            flight_info = flight.get("flights", [{}])[0]
            all_flights_data[str(idx)] = {
                "airline": flight_info.get("airline", "Unknown Airline"),
                "flight_number": flight_info.get("flight_number", "N/A"),
                "cabin_class": flight_info.get("travel_class", "N/A"),
                "aircraft": flight_info.get("airplane", "N/A"),
                "price": flight.get("price", "N/A"),
                "departure_time": flight_info.get("departure_airport", {}).get("time", "N/A"),
                "arrival_time": flight_info.get("arrival_airport", {}).get("time", "N/A"),
                "duration": flight_info.get("duration", "N/A"),
                "layovers": ', '.join(
                    f"{lay.get('name', lay.get('id', 'Unknown'))} ({lay.get('duration', 'N/A')} min)"
                    for lay in flight.get("layovers", [])
                ) or "None"
            }
        return all_flights_data
    except Exception as e:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id="flight_tool",
            details={"error": str(e)},
            severity="error"
        )
        return {"error": str(e)}


@tool
def estimate_hotel_cost(city: str, check_in: str, check_out: str) -> dict:
    """Use SerpAPI to get hotel listings and fetch full addresses."""
    api_key = os.getenv("SERPAPI_API_KEY")
    try:
        hotel_search = GoogleSearch({
            "engine": "google_hotels",
            "q": f"hotels in {city}",
            "check_in_date": check_in,
            "check_out_date": check_out,
            "currency": "USD",
            "hl": "en",
            "domain": "google.com",
            "api_key": api_key
        })

        hotel_results = hotel_search.get_dict()
        hotels = hotel_results.get("hotels") or hotel_results.get("properties") or hotel_results.get("property_results")

        all_hotels_data = {}
        for idx, hotel in enumerate(hotels[:5], start=1):
            name = hotel.get("name", "Unknown Hotel")
            rating = hotel.get("overall_rating", hotel.get("rating", "N/A"))
            stars = hotel.get("hotel_class", hotel.get("stars", "N/A"))
            check_in_time = hotel.get("check_in_time", "N/A")
            check_out_time = hotel.get("check_out_time", "N/A")
            price = hotel.get("rate_per_night", {}).get("before_taxes_fees") or \
                    hotel.get("price", {}).get("displayed_price", "N/A")
            website = hotel.get("link", "Not available")
            amenities = hotel.get("amenities", [])
            top_amenities = ", ".join(amenities[:5]) if amenities else "Not listed"
            reviews_raw = hotel.get("reviews", {})
            if isinstance(reviews_raw, dict):
                reviews = reviews_raw.get("count", hotel.get("reviews_count", "N/A"))
            else:
                reviews = hotel.get("reviews_count", "N/A")


            all_hotels_data[str(idx)] = {
                "name": name,
                "stars": stars,
                "rating": rating,
                "reviews": reviews,
                "check_in": check_in_time,
                "check_out": check_out_time,
                "price_per_night": price,
                "top_amenities": top_amenities,
                "website": website
            }

        return all_hotels_data
    except Exception as e:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id="hotel_tool",
            details={"error": str(e)},
            severity="error"
        )
        return {"error": str(e)}


@tool
def get_weather_forecast(city: str, days: int = 7) -> str:
    """
    Get weather forecast (up to 14 days) for a given city using WeatherAPI.com.
    Returns a formatted string of daily forecasts.
    """
    api_key = os.getenv("WEATHERAPI_KEY")
    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": city,
        "days": min(days, 14),  # API only supports up to 14 days
        "aqi": "no",
        "alerts": "no"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        forecast_days = data.get("forecast", {}).get("forecastday", [])
        if not forecast_days:
            return "Forecast unavailable."

        forecast_lines = []
        for day in forecast_days:
            date = day["date"]
            condition = day["day"]["condition"]["text"]
            max_temp = day["day"]["maxtemp_c"]
            min_temp = day["day"]["mintemp_c"]
            forecast_lines.append(
                f"{date}: High {max_temp}C / Low {min_temp}C - {condition}"
            )

        return "\n".join(forecast_lines)

    except Exception as e:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id="weather_tool",
            details={"error": str(e)},
            severity="error"
        )
        return f"Error retrieving weather: {str(e)}"



@tool
def search_places(query: str) -> str:
    """Search tourist attractions using SerpAPI."""
    try:
        response = requests.get("https://serpapi.com/search", params={
            "engine": "google",
            "q": f"Top tourist attractions in {query}",
            "hl": "en",
            "gl": "us",
            "num": 5,
            "api_key": os.getenv("SERPAPI_API_KEY")
        })
        data = response.json()
        results = data.get("organic_results", [])
        return "\n".join(
            f"* {place.get('title')}\n_{place.get('snippet')}_\n[More info]({place.get('link')})"
            for place in results
        )
    except Exception as e:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id="places_tool",
            details={"error": str(e)},
            severity="error"
        )
        return "Error fetching places."


# ---------------- PROMPT CHAINS (using hardened templates) -------------------

info_extraction_prompt = HardenedPromptTemplates.get_info_extraction_prompt()
extract_info_chain = info_extraction_prompt | llm | parser

refine_prompt = HardenedPromptTemplates.get_refine_prompt()
refine_chain = refine_prompt | llm | parser


#----------------------- Daily Spending in a City -----------------------

@tool
def get_cost_breakdown(city: str, preferences: str) -> str:
    """Use SerpAPI to extract breakdown of travel costs (restaurants, attractions, transport, shopping) for low-budget, mid-range or luxury tourists."""
    try:
        query = f"average daily travel cost breakdown for {preferences} tourists in {city} including food, transport, attractions, shopping"
        response = requests.get("https://serpapi.com/search", params={
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 1,
            "hl": "en"
        })
        results = response.json().get("organic_results", [])
        for result in results:
            snippet = result.get("snippet")
            if snippet:
                return snippet
    except Exception as e:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id="cost_tool",
            details={"error": str(e)},
            severity="error"
        )
        return f"Could not fetch cost breakdown due to error: {str(e)}"

    return "No detailed cost breakdown found for the specified city and travel preference."


misc_cost_prompt = HardenedPromptTemplates.get_misc_cost_prompt()
misc_cost_chain = misc_cost_prompt | llm | StrOutputParser()



def calculate_total_budget(state: TravelPlannerState, serpapi_snippet: str) -> str:
    import re
    from statistics import mean

    # Days of travel
    start = datetime.datetime.strptime(state["outbound_date"], "%Y-%m-%d")
    end = datetime.datetime.strptime(state["return_date"], "%Y-%m-%d")
    days = max((end - start).days, 1)

    # 1. Flight range
    flight_prices = [
        float(str(f.get("price", "")).replace("$", "").replace(",", ""))
        for f in state.get("flight", {}).values()
        if f.get("price") is not None
    ]
    min_flight = min(flight_prices) if flight_prices else 0
    max_flight = max(flight_prices) if flight_prices else 0
    avg_flight = mean(flight_prices) if flight_prices else 0

    # 2. Hotel range
    hotel_prices = [
        float(str(h.get("price_per_night", "")).replace("$", "").replace(",", ""))
        for h in state.get("hotel", {}).values()
        if h.get("price_per_night") is not None
    ]
    min_hotel = min(hotel_prices) if hotel_prices else 0
    max_hotel = max(hotel_prices) if hotel_prices else 0
    avg_hotel = mean(hotel_prices) if hotel_prices else 0
    hotel_total = avg_hotel * days

    llm_response = misc_cost_chain.invoke({
        "snippet": HardenedPromptTemplates.sanitize_for_prompt(serpapi_snippet),
        "preference": state.get("preferences", "mid-range")
    })

    # Extract daily cost from LLM response
    try:
        cost_match = re.search(r"\$([0-9,]+)", llm_response)
        daily_misc = float(cost_match.group(1).replace(',', '')) if cost_match else 100
    except (AttributeError, ValueError):
        daily_misc = 100  # Default fallback

    # Final message string
    message = f"""\
    Flight Cost Range: ${min_flight} - ${max_flight}
    Hotel (per night): ${min_hotel} - ${max_hotel}
    Trip Length: {days} nights

    Hotel Total: ${hotel_total}
    Miscellaneous Daily Cost: {llm_response}
    Total Estimated Cost: ${min_flight + hotel_total + days * daily_misc} - ${max_flight + hotel_total + days * daily_misc}
    """
    return message



# ---------------- ITINERARY GENERATION ----------------

itinerary_prompt = HardenedPromptTemplates.get_itinerary_prompt()
itinerary_chain = itinerary_prompt | llm | parser

# ---------------- GRAPH NODES ----------------

def security_gate(state: TravelPlannerState) -> TravelPlannerState:
    """
    Security gate node - runs security checks on user input.
    This is the entry point of the graph.
    """
    session_id = state.get("session_id", "unknown")
    user_input = state.get("user_input", "")

    # Run security middleware checks
    security_result = security_middleware.check(
        user_input=user_input,
        session_id=session_id,
        skip_topic_check=False  # Enforce travel-related topics
    )

    if security_result.is_allowed:
        state["security_passed"] = True
        state["user_input"] = security_result.sanitized_input
        security_logger.log_security_gate(session_id, passed=True)
    else:
        state["security_passed"] = False
        state["security_block_reason"] = security_result.block_reason
        security_logger.log_security_gate(
            session_id,
            passed=False,
            block_reason=security_result.block_reason
        )

    return state


def blocked_response(state: TravelPlannerState) -> TravelPlannerState:
    """
    Handle blocked requests with a safe response.
    """
    block_reason = state.get("security_block_reason", "Security check failed")
    state["itinerary"] = f"I'm sorry, but I couldn't process your request. {block_reason}. Please try rephrasing your travel query."
    return state


def extract_info(state: TravelPlannerState) -> TravelPlannerState:
    """Extract travel information from user input using safe parsing."""
    session_id = state.get("session_id", "unknown")

    # Sanitize input for prompt
    sanitized_input = HardenedPromptTemplates.sanitize_for_prompt(state["user_input"])
    result = extract_info_chain.invoke({"input": sanitized_input})

    # Use safe parser instead of eval()
    parse_result = safe_parser.parse_json(result, session_id)

    if parse_result.success and parse_result.data:
        # Validate against schema
        is_valid, validated, error = InputValidator.validate_extracted_info(parse_result.data)

        if is_valid and validated:
            info = validated.model_dump()
        else:
            security_logger.log_event(
                event_type=SecurityEventType.OUTPUT_VALIDATION,
                session_id=session_id,
                details={"validation_error": error},
                severity="warning"
            )
            info = parse_result.data  # Use unvalidated data as fallback
    else:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id=session_id,
            details={"parse_error": parse_result.error},
            severity="warning"
        )
        info = {}

    for key in ["departure_city", "destination_country", "preferences", "budget", "outbound_date", "return_date"]:
        state[key] = info.get(key, "") or state.get(key, "")
    return state


def plan_node(state: TravelPlannerState) -> TravelPlannerState:
    """Refine travel plan using safe parsing."""
    session_id = state.get("session_id", "unknown")
    result = refine_chain.invoke(state)

    # Use safe parser instead of eval()
    parse_result = safe_parser.parse_json(result, session_id)

    if parse_result.success and parse_result.data:
        # Validate against schema
        is_valid, validated, error = InputValidator.validate_refined_info(parse_result.data)

        if is_valid and validated:
            update = validated.model_dump()
        else:
            security_logger.log_event(
                event_type=SecurityEventType.OUTPUT_VALIDATION,
                session_id=session_id,
                details={"validation_error": error},
                severity="warning"
            )
            update = parse_result.data
    else:
        security_logger.log_event(
            event_type=SecurityEventType.OUTPUT_VALIDATION,
            session_id=session_id,
            details={"parse_error": parse_result.error},
            severity="warning"
        )
        update = {}

    state["destination_city"] = update.get("destination_city", state["destination_country"])
    state["arrival_airport"] = update.get("arrival_airport", "")
    state["notes"] = update.get("notes", "")
    return state


def flight_step(state: TravelPlannerState) -> TravelPlannerState:
    state["flight"] = estimate_flight_cost.invoke({
        "departure_city": state["departure_city"],
        "destination_city": state["destination_city"],
        "outbound_date": state["outbound_date"],
        "return_date": state["return_date"]
    })
    return state


def hotel_step(state: TravelPlannerState) -> TravelPlannerState:
    state["hotel"] = estimate_hotel_cost.invoke({
    "city": state["destination_city"],
    "check_in": state["outbound_date"],
    "check_out": state["return_date"]
})

    return state



def budget_step(state: TravelPlannerState) -> TravelPlannerState:
    snippet = get_cost_breakdown.invoke({
        "city": state["destination_city"],
        "preferences": state["preferences"]
    })

    state["total"] = calculate_total_budget(state, snippet)
    return state




def weather_step(state: TravelPlannerState) -> TravelPlannerState:
    start = datetime.datetime.strptime(state["outbound_date"], "%Y-%m-%d")
    end = datetime.datetime.strptime(state["return_date"], "%Y-%m-%d")
    days = (end - start).days
    state["weather"] = get_weather_forecast.invoke({
        "city": state["destination_city"],
        "days": days
    })
    return state


def search_step(state: TravelPlannerState) -> TravelPlannerState:
    state["places"] = search_places.invoke({"query": state["destination_city"]})
    return state


def itinerary_step(state: TravelPlannerState) -> TravelPlannerState:
    """Generate itinerary with PII detection."""
    session_id = state.get("session_id", "unknown")

    itinerary = itinerary_chain.invoke(state)

    # Check for PII in output
    pii_result = pii_detector.detect(itinerary, session_id)
    if pii_result.contains_pii:
        security_logger.log_pii_detection(
            session_id=session_id,
            entity_types=[e.entity_type for e in pii_result.entities],
            count=len(pii_result.entities)
        )
        # Use anonymized version
        itinerary = pii_result.anonymized_text or itinerary

    state["itinerary"] = itinerary
    return state


# ---------------- LANGGRAPH ----------------

def route_after_security(state: TravelPlannerState) -> str:
    """Route based on security check result."""
    if state.get("security_passed", False):
        return "extract_info"
    else:
        return "blocked"


graph = StateGraph(TravelPlannerState)

# Add security gate as entry point
graph.add_node("security_gate", security_gate)
graph.add_node("blocked", blocked_response)
graph.add_node("extract_info", extract_info)
graph.add_node("plan_node", plan_node)
graph.add_node("flight_step", flight_step)
graph.add_node("hotel_step", hotel_step)
graph.add_node("budget_step", budget_step)
graph.add_node("weather_step", weather_step)
graph.add_node("search_step", search_step)
graph.add_node("itinerary_step", itinerary_step)

# Set security gate as entry point
graph.set_entry_point("security_gate")

# Add conditional routing after security check
graph.add_conditional_edges(
    "security_gate",
    route_after_security,
    {
        "extract_info": "extract_info",
        "blocked": "blocked"
    }
)

# Blocked requests go to END
graph.add_edge("blocked", END)

# Normal flow
graph.add_edge("extract_info", "plan_node")
graph.add_edge("plan_node", "flight_step")
graph.add_edge("flight_step", "hotel_step")
graph.add_edge("hotel_step", "budget_step")
graph.add_edge("budget_step", "weather_step")
graph.add_edge("weather_step", "search_step")
graph.add_edge("search_step", "itinerary_step")
graph.add_edge("itinerary_step", END)

travel_graph = graph.compile()
