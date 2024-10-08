# Common imports
import os
from dotenv import load_dotenv
#import json
from openai import OpenAI
import streamlit as st
# import lolviz

# Import the key CrewAI classes
from crewai import Agent, Task, Crew

if load_dotenv('.env'):
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
    OPENAI_KEY = st.secrets['OPENAI_API_KEY']

client = OpenAI(api_key=OPENAI_KEY)


# Create a new instance of the WebsiteSearchTool
# Set the base URL of a website, e.g., "https://example.com/", so that the tool can search for sub-pages on that website

from crewai_tools import WebsiteSearchTool
tool_websearch = WebsiteSearchTool("https://www.moe.gov.sg/post-secondary/admissions")

# Agents
agent_admission = Agent(
    role="Admission Advisor",

    goal="Provide clear and targetted advice on post secondary school admission criteria and procedure specific to the client as specified by his/her input: {topic}.",

    backstory="""You are an expert in Singapore's Minstry of Education Post Secondary School registration and admission matters.
    You will read and understand the admission criteria, procedures and timelines for various post secondary school education options
    and their respective applicable admission exercises from the Minstry of Education's website using the task tool. \
    You will then analyze the input provided by the client: '{topic}', \
    and advise the client helpfully and factually. Provide all the important information that he/she needs to know to be successful in the application.
    If you are providing procedure steps in your response, make them clear and easy to follow with key timeline to take note of.
    Make use of the tool provided to source for answers to questions on Singapore's Minstry of Education Post Secondary School registration and admission matters.""",

    allow_delegation=False, # we will explain more about this later

	  verbose=True, # to allow the agent to print out the steps it is taking

    tools=[tool_websearch],
)

# Tasks
task_admission = Task(
    description="""\
    1. Understand all the admission exercises detailed in MOE website.
    2. Identify the suitable admission exercise(s) based on the inputs provided by the client: {topic}.
    3. Provide useful and factual answer(s) to the client based on the identified admission exercise(s).
    4. Do not provide information on any admission exercises that the client is ineligible for or which timeline has already passed.
    5. If the client is asking for the next step(s) in the process or what he/she needs to do next, \
    provide useful and simple-to-follow step-by-step guide and important timeline with checkboxes on how to proceed in the admission procedure.
    6. If the input is generic and does not provide enough information for you to provide useful, practical response(s) regarding post-secondary school admission exercise(s), \
    ask the user for more specific information.
    7. If the information provided about the client in the input does not meet the criteria of all admission exercises, \
    be encouraging and provide practical advice what the client can do for further education. Provide helpful timeline.
    8. Always provide the exact url links to the source of information as reference to the client for each point of suggestion or advice or guide you provide in your response.
    """,

    expected_output="""\
    Useful and easy to understand information about suitable Post-Secondary school admission exercise(s) based on the user's question/prompt. \
    Include clear instructions for the next step(s) in the application / registration procedure where required.""",

    agent=agent_admission,
)

# Crew
crew = Crew(
    agents=[agent_admission],
    tasks=[task_admission],
    verbose=True
)

# Get current date
from datetime import date
# Kickoff
# Start the crew's task execution
def crew_kickoff(user_message):
    today = date.today()
    result = crew.kickoff(inputs={"topic": f"{user_message} Today's date is {today}"})
    return result