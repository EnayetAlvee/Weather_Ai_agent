import os
import requests
import streamlit as st
from datetime import datetime, timedelta
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import re

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Initialize LLM model (ChatGroq)
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.5,
)

# Function to fetch the location by IP
@st.cache_data(ttl=600)
def get_location_by_ip():
    try:
        ip_address = requests.get('https://api.ipify.org').text
        url = f"https://ipinfo.io/{ip_address}/json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            location = data.get("city")
            return location
    except Exception:
        return None

# Function to fetch weather data by city name
@st.cache_data(ttl=600)
def get_weather(location, date=None):
    if date is None:
        # Use /weather endpoint for real-time current weather
        url = "https://api.openweathermap.org/data/2.5/weather"
    else:
        # Use forecast endpoint for future or specific date weather
        url = "https://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if date:
            # Find the weather for the given date in the forecast data
            for forecast in data["list"]:
                forecast_time = datetime.utcfromtimestamp(forecast["dt"])
                if forecast_time.date() == date.date():
                    weather = {
                        "location": data["city"]["name"],
                        "temperature": forecast["main"]["temp"],
                        "description": forecast["weather"][0]["description"],
                        "icon": forecast["weather"][0]["icon"],
                        "humidity": forecast["main"]["humidity"],
                        "pressure": forecast["main"]["pressure"],
                        "wind_speed": forecast["wind"]["speed"],
                        "date": forecast_time.strftime('%Y-%m-%d'),
                    }
                    return weather
            return {"error": "Weather data not available for this date."}
        else:
            # Return weather for the current day from /weather endpoint
            weather = {
                "location": data["name"],
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "date": datetime.utcfromtimestamp(data["dt"]).strftime('%Y-%m-%d'),
            }
            return weather
    else:
        return {"error": f"API error: {response.status_code}"}

# Function to format weather data into a string
# Function to format weather data into a detailed string
def format_weather_response(weather, days_ahead=None):
    if not weather or "error" in weather:
        return weather.get("error", "Could not fetch weather data.")
    
    # Basic weather details for the concise format
    weather_details = (
        f"Temperature: {weather['temperature']}°C, "
        f"Weather: {weather['description']}, "
        f"Humidity: {weather['humidity']}%, "
        f"Pressure: {weather['pressure']} hPa, "
        f"Wind Speed: {weather['wind_speed']} m/s"
    )
    
    # Detailed narrative description
    description = ""
    if "haze" in weather['description'].lower():
        description = (
            f"The weather in {weather['location']} on {weather['date']} will be characterized by {weather['description']} with a temperature of {weather['temperature']}°C. "
            f"The humidity is at {weather['humidity']}%, making the air feel quite heavy and sticky, typical for this region. "
            f"Atmospheric pressure is {weather['pressure']} hPa, suggesting stable conditions, though the haze indicates potential air quality concerns, possibly due to pollution or dust. "
            f"A gentle breeze at {weather['wind_speed']} m/s offers slight relief but isn't strong enough to clear the haze. "
            f"Light clothing and staying hydrated are recommended, and consider limiting outdoor activities if you're sensitive to air quality."
        )
    elif "cloud" in weather['description'].lower():
        description = (
            f"The weather in {weather['location']} on {weather['date']} will feature {weather['description']} with a temperature of {weather['temperature']}°C. "
            f"The cloud cover will provide some shade, offering relief from direct sunlight and making it feel more comfortable compared to clearer days. "
            f"With humidity at {weather['humidity']}%, the air might feel a bit damp, and the pressure at {weather['pressure']} hPa suggests a stable atmosphere. "
            f"Wind speed is {weather['wind_speed']} m/s, which is mild and won’t significantly impact the day. "
            f"The cloudy conditions might hint at a chance of light rain or drizzle, so carrying an umbrella could be wise."
        )
    else:
        description = (
            f"The weather in {weather['location']} on {weather['date']} will be {weather['description']} with a temperature of {weather['temperature']}°C. "
            f"Humidity is at {weather['humidity']}%, pressure at {weather['pressure']} hPa, and wind speed at {weather['wind_speed']} m/s. "
            f"Expect typical conditions for this weather pattern—stay prepared for changes and dress accordingly."
        )

    # If this is a future date query (e.g., "after 3 days"), format the response as requested
    if days_ahead is not None:
        date_obj = datetime.strptime(weather['date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%d %B')
        return f"the weather after {days_ahead} days in {formatted_date} will be like {{ {weather_details} }}\n\n{description}"

    # Default format for current or other date queries
    return f"{description}"
# Function to extract explicit dates in YYYY-MM-DD format
def extract_date_from_query(query):
    """Extract explicit date in YYYY-MM-DD format from the user's query. Let the agent handle natural language dates."""
    date_pattern = r"\b(\d{4}-\d{2}-\d{2})\b"
    match = re.search(date_pattern, query)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d")
        except Exception:
            return None
    return None

# Function to fetch weather without providing location
def get_weather_without_location(query, date=None):
    # If no date is provided by the agent, default to today
    if date is None:
        date = datetime.utcnow()

    location = get_location_by_ip()
    if location:
        # Calculate days ahead if the date is in the future
        days_ahead = (date.date() - datetime.utcnow().date()).days if date > datetime.utcnow() else None
        weather = get_weather(location, date)
        return format_weather_response(weather, days_ahead)
    else:
        return "Could not determine location by IP."

# Function to fetch weather for a specific location
def get_weather_for_location(location, date=None):
    # If no date is provided, default to today
    if date is None:
        date = datetime.utcnow()
    
    # Calculate days ahead if the date is in the future
    days_ahead = (date.date() - datetime.utcnow().date()).days if date > datetime.utcnow() else None
    weather = get_weather(location, date)
    return format_weather_response(weather, days_ahead)

# Define tools for the agent
get_weather_by_ip = Tool(
    name="get_weather_by_ip",
    func=lambda query: get_weather_without_location(query, extract_date_from_query(query)),
    description="Get weather for a location based on user's IP. The agent should interpret natural language dates (e.g., 'today', 'tomorrow', 'next week', 'in 3 days') relative to May 23, 2025, and pass the date to this tool. If no date is specified, it defaults to the current date."
)

get_current_weather = Tool(
    name="get_current_weather",
    func=lambda location: get_weather_for_location(location, extract_date_from_query(location)),
    description="Get weather for a specific location. Input should be a city name. The agent should interpret natural language dates (e.g., 'today', 'tomorrow', 'next week', 'in 3 days') relative to May 23, 2025, and pass the date to this tool. If no date is specified, it defaults to the current date."
)

# Initialize the agent with chat history support
tools = [get_current_weather, get_weather_by_ip]

system_prompt = """
You are a helpful assistant that provides weather information using the OpenWeatherMap API. You can answer questions about current weather and forecasts. Use the tools to fetch weather data.

- If the user doesn't specify a location, use the get_weather_by_ip tool to determine the location via IP and fetch the weather.
- If the user specifies a location, use the get_current_weather tool with the provided location.
- The current date is May 23, 2025. You must interpret natural language date expressions in the user's query (e.g., 'today', 'tomorrow', 'yesterday', 'next week', 'in 3 days', '2 months ago', 'next year') and convert them to actual dates relative to May 23, 2025. For example:
  - 'today' is May 23, 2025
  - 'tomorrow' is May 24, 2025
  - 'yesterday' is May 22, 2025
  - 'next week' is May 30, 2025
  - 'in 3 days' is May 26, 2025
  - 'after 3 days' is May 26, 2025
  - '2 months ago' is March 23, 2025
  - 'next year' is May 23, 2026
- After interpreting the date, pass the date to the appropriate tool (get_current_weather or get_weather_by_ip) to fetch the weather for that date. If the date is not specified, assume the current date (May 23, 2025).
- If the location is a country (e.g., US, Bangladesh) or a large city (e.g., New York, Dhaka), append the following warning to your response: "\n**Warning:** This is a country or large city. Weather may vary by specific location within this region." Use your knowledge to determine if the location fits this category.
- Respond with the weather information as provided by the tool, ensuring the warning is included when applicable.
- Provide clear, concise weather information, including temperature, weather description, humidity, pressure, and wind speed.
"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    system_message=system_prompt,
    handle_parsing_errors=True,
)
# Streamlit application for user input and output display
st.title("Weather Query Agent")

st.write("This is an AI-powered weather assistant. You can ask about the current weather and forecasts.")

# Initialize chat history for agent's memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# User input for the weather query
user_query = st.text_input("Ask a weather-related question:", value="")

if user_query:
    with st.spinner("Getting response..."):
        # Run the agent with the user query
        response = agent.invoke(user_query, handle_parsing_errors=True)
        # Use the tool's output directly if available, falling back to the agent's response
        tool_output = response.get("output", response) if isinstance(response, dict) else response
        # Save chat history for agent's memory
        st.session_state.chat_history.append({"user": user_query, "agent": tool_output})

    st.write("**Agent Response:**")
    st.write(tool_output)
else:
    st.write("Please enter a weather-related query.")




