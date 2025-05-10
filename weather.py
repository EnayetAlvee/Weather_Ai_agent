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

# Function to fetch weather data by city name
def get_weather(location, date=None):
    url = f"http://api.openweathermap.org/data/2.5/forecast?"
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
                        "humidity": forecast["main"]["humidity"],  # Humidity in percentage
                        "pressure": forecast["main"]["pressure"],  # Atmospheric pressure in hPa
                        "wind_speed": forecast["wind"]["speed"],   # Wind speed in m/s
                        "date": forecast_time.strftime('%Y-%m-%d'),
                    }
                    return weather
            return {"error": "Weather data not available for this date."}
        else:
            # Return weather for the current day
            weather = {
                "location": data["city"]["name"],
                "temperature": data["list"][0]["main"]["temp"],
                "description": data["list"][0]["weather"][0]["description"],
                "icon": data["list"][0]["weather"][0]["icon"],
                "humidity": data["list"][0]["main"]["humidity"],
                "pressure": data["list"][0]["main"]["pressure"],
                "wind_speed": data["list"][0]["wind"]["speed"],
            }
            return weather
    else:
        return None

# Function to fetch the location by IP
def get_location_by_ip():
    ip_address = requests.get('https://api.ipify.org').text
    url = f"https://ipinfo.io/{ip_address}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        location = data["city"]
        return location
    else:
        return None

# Function to fetch weather without providing location
def get_weather_without_location(query, date=None):
    location = get_location_by_ip()
    if location:
        weather = get_weather(location, date)
        return weather
    else:
        return None

# Define tools for the agent
weather_tool_without_location = Tool(
    name="WeatherWithoutLocation",
    func=get_weather_without_location,
    description="Get current weather for a location without specifying the city name."
)

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get current weather for a location. Input should be a city name."
)

# Initialize the agent
tools = [weather_tool, weather_tool_without_location]

system_prompt = """
You are a helpful assistant that provides weather information. You can answer questions about current weather and forecasts using the OpenWeatherMap API. You can also extract locations from user queries and provide answers based on the weather data you retrieve.  
You are also able to use the tools Weather and WeatherWithoutLocation to get current weather data. Now answer the user's question using the tools if necessary.
"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit application for user input and output display
st.title("Weather Query Agent")

st.write("This is an AI-powered weather assistant. You can ask about the current weather and forecasts.")

# User input for the weather query
user_query = st.text_input("Ask a weather-related question:")

def extract_date_from_query(query):
    """Extract date from the user's query if available."""
    date_pattern = r"\b(\d{4}-\d{2}-\d{2})\b"  # regex for 'YYYY-MM-DD' format
    match = re.search(date_pattern, query)
    if match:
        return datetime.strptime(match.group(0), "%Y-%m-%d")
    return None

if user_query:
    # Extract date from user query if present
    date = extract_date_from_query(user_query)

    with st.spinner("Getting response..."):
        # Pass the specific date to the agent if present, else use current date
        response = agent.invoke(system_prompt + f"Check the weather for {user_query}", handle_parsing_errors=True)

        st.write("Agent Response:")
        if response.get("output"):
            # Display detailed weather information
            weather_data = response["output"]
            if isinstance(weather_data, dict):
                st.write(f"**Location**: {weather_data['location']}")
                st.write(f"**Temperature**: {weather_data['temperature']}°C")
                st.write(f"**Weather**: {weather_data['description']}")
                st.write(f"**Humidity**: {weather_data['humidity']}%")
                st.write(f"**Pressure**: {weather_data['pressure']} hPa")
                st.write(f"**Wind Speed**: {weather_data['wind_speed']} m/s")
                st.write(f"**Date**: {weather_data['date']}")
                st.image(f"http://openweathermap.org/img/wn/{weather_data['icon']}@2x.png")
            else:
                st.write(weather_data)
        else:
            st.write("Could not fetch weather data. Please try again.")
else:
    st.write("Please enter a weather-related query.")
