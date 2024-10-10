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
tool_admission_websearch = WebsiteSearchTool("https://www.moe.gov.sg/post-secondary/admissions",
                                             temperature=0)
tool_ecg_websearch = WebsiteSearchTool("https://www.moe.gov.sg/coursefinder",
                                             temperature=0)


# Agents
agent_friendly_advisor = Agent(
    role="Post-Secondary Education Advisor",

    goal="Consolidate the responses provided by the other agents, \
        and rewrite a coherent, helpful and succinct final response to the user's query.",

    backstory=f"""You are a superb communicator whose field of expertise is in Singapore's Post-Secondary School Education. 
    Your job is to consolidate the information in the responses provided by the other agents, \
        and communicate information that is coherent and only relevant to the user's query. 
        You will also craft your resonse in such a way that is easy for a Post Secondary School student to understand. 
        The vast majority of Post Secondary School students are between 16 and 17 years old. 
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)

agent_admission = Agent(
    role="Admission Advisor",

    goal="Provide clear and targetted advice on post secondary school education admission criteria and procedure \
        specific to the user's query: {topic_admission}.",

    backstory="""You are an expert in Singapore's Minstry of Education Post Secondary School registration and admission matters.
    You have the capability to read and understand all admission matters on post secondary school education and the various admission exercises, \
        such as admission criteria, procedures and timelines. 
    You are also capable of analyze the user's query and provide a factual and useful response. 
    Your work will be the basis for the Post-Secondary Education Advisor to craft the final response to the user's query.
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)

agent_ecg = Agent(
    role="Education and Career Advisor",

    goal="Perform Values, Interests, Personality, Skills (V.I.P.S.) analysis of the user based on the information \
        provided by the user in his/her query: {topic_ecg}.\
        Shortlist 5 Post-Secondary School Education courses/schools that are most suitable to the user based on the V.I.P.S analysis.",

    backstory="""You are an expert in identifying Post-Secondary School Education courses/schools that best-match the V.I.P.S. analysis of students.\
          Singapore's Minstry of Education Post Secondary School registration and admission matters.
          Your work will be the basis for the Post-Secondary Education Advisor to craft the final response to the user's query.
          """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking

    tools=[tool_ecg_websearch],
)

# Tasks
task_craft_response = Task(
    description="""\
    1. Analyse the work produced by the other agents 
    2. Consolidate and compile the work provided by the other agents in a logical, coherent and meaningful way. 
    3. From the output of the previous step, craft your response using ONLY information that is applicable to the user based on his/her query. 
    4. Craft your response using this information in a manner that is succinct and easy to follow, suitable for Post-Secondary School students.
    """,

    expected_output="""\
    Factual, useful and coherent information/advice only relevant to the user's query on matters about Post-Secondary Education \
        crafted in a way that is succinct and easy to understand.
    """,

    agent=agent_friendly_advisor,

    # tools=[tool_admission_websearch],
)

task_admission = Task(
    description="""\
    1. Understand all admission matters on Post Secondary School Education.
    2. Analyse information provided by the user in his/her query: {topic_admission}.
    3. Respond factually and meaningfully to the user. 
    4. If the user is asking about eligibility, respond with only information of admission exercises and options that the user is eligible for.\
    Ask for more information if the user did not provide enough information for you to answer the query.  
    5. Check the timeline(s). Highlight if and when the eligible admission exercise(s) is/are open. 
    6. Include the url link(s) to the webpage(s) where the information in your response are sourced from.
    """,

    expected_output="""\
    Factual, useful and succinct information/advice on matters about Post-Secondary Education admission exercises in response to the user's query.
    """,

    agent=agent_admission,

    tools=[tool_admission_websearch],
)

# Tasks
task_ecg = Task(
    description="""\
    1. Perform Values, Interests, Personality, Skills (V.I.P.S.) analysis of the user based on the information \
        provided by the user in his/her query: {topic_ecg}.
    2. Shortlist not more than 5 Post-Secondary Education only courses/schools suitable for the user based on the analysis.
    3. Do NOT include University courses.
    3. Rank the shortlisted courses/schools with the most suitable one first and the least suitable last. 
    4. Include the url link(s) to the webpage(s) of the identified courses/schools in your list.
    """,

    expected_output="""\
    A list 5 Post-Secondary Education only courses/schools suitable for the user based on the V.I.P.S. analysis. \
        """,

    agent=agent_ecg,

)

# Crew
crew = Crew(
    agents=[agent_admission, agent_ecg, agent_friendly_advisor],
    tasks=[],
    verbose=True
)

# Get current date
from datetime import date
# Kickoff
# Start the crew's task execution
def crew_kickoff(dict_component_queries):
    today = date.today()
    result = crew.kickoff(inputs={"topic_admission": f"{dict_component_queries.get('Admission')}. Today's date is {today}",
                                  "topic_ecg":f"{dict_component_queries.get('ECG')}"})
    return result

def let_the_agents_handle_it(dict_component_queries):
    if "ECG" in dict_component_queries.keys():
        crew.tasks.append(task_ecg)
    if "Admission" in dict_component_queries.keys():
        # crew.agents.append(agent_admission)
        crew.tasks.append(task_admission)
    if len(crew.tasks)==0:
        response = ""
    else:
        crew.tasks.append(task_craft_response)
        response = crew_kickoff(dict_component_queries).raw
    return response
