import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os
from agents.tool import function_tool
import requests

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
api_key = os.getenv("OPENWEATHER_API_KEY") 

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


external_client = AsyncOpenAI( 
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

@function_tool("get_weather")
def get_weather(Location: str, Unit: str = "Celsius") -> str:

    """Fetch current weather for a given city using OpenWeatherMap API, return the weather.
    """ 
    
    unit_system = "metric" if Unit.lower() == "celsius" else "imperial"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={Location}&appid={api_key}&units={unit_system}"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return f"Couldn't fetch weather for {Location}. Reason: {data.get('message', 'Unknown error')}"

        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]

        return f"The weather in {Location} is currently {weather} with a temperature of {temperature}°{Unit}, feels like {feels_like}°{Unit}."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


agent = Agent(
    name = "teacher/tutor assistant",
    instructions = "you explain concepts about AI Agent in a simple and easy to understand way, you are a teacher/tutor assistant about AI Agent. You are very patient and helpful. You always answer the question asked by the user, and if you don't know the answer, you say that you don't know the answer. if someone asks about weather then use the get_weather tool to get the weather You never say that you are an AI model or an agent, you always act like a human being.",
    model=model,
    tools=[get_weather]
    )

@cl.on_chat_start
async def handle_chat_start():

    cl.user_session.set("history", []) 

    await cl.Message(
        content="Hello! how can i help you today?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history") 

    history.append(
        {"role": "user", "content": message.content}
    )


    AgentResult = await cl.make_async(Runner.run_sync)(agent, input=history)

    final_response = AgentResult.final_output
    await cl.Message(content=final_response).send()

    history.append({"role": "assistant", "content": final_response})
    cl.user_session.set("history", history)

