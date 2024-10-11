# Common imports
import os
from dotenv import load_dotenv
import json
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

    goal="""Work with othe expert agents to craft a targetted and helpful response to user's query on Post-Secondary School Education.\
        The query has been analysed and broken down into categorised prompts in the format of {{'category':'prompts'}}.\
        User's query: {topics}""" ,

    backstory=f"""You are a superb communicator whose field of expertise is in Singapore's Post-Secondary School Education. 
        Your job is to work with other agents in your crew to provide factual, targetted and useful information/advice \
        to the user's query. 
        Craft your resonse in such a way that is succinct and easy for a Post Secondary School student to understand. 
        The vast majority of Post Secondary School students are between 16 and 17 years old. 
    """,

    allow_delegation=True, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)

agent_admission = Agent(
    role="Admission Expert",

    goal="Provide clear, factual and targetted information/advice on post secondary school education admission criteria and procedure \
        specific to the user's prompt in the 'Admissions' category.",

    backstory="""You are an expert in Singapore's Minstry of Education Post Secondary School admission matters.
    You will analyse user's prompt(s) regarding admission matters, research about it on the Singapore MOE Post Secondary School Admissions website,\
        and provide factual and targetted information/advice to the Post-Secondary Education Advisor. 
    Your work will be the basis for the Post-Secondary Education Advisor to craft the final response to the user.
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)

agent_ecg = Agent(
    role="Education and Career Guidance Expert",

    goal="Perform Values, Interests, Personality, Skills (VIPS) analysis of the user based on the user's prompt in the 'ECG' category of the query. \
        Identify Post-Secondary School Education courses/schools that will suit to the user based on the VIPS analysis.",

    backstory="""You are an Education and Career Guidance expert. 
            Your skills are in identifying Post-Secondary courses/schools suitable to the user based on his/her VIPS. 
            Your work will be the basis for the Post-Secondary Education Advisor to craft the final response to the user's query.
          """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking

    tools=[tool_ecg_websearch],
)

# Tasks
task_craft_response = Task(
    description="""\
    1. Understand the user's query, which has been broken down into component prompts and categorised in the format {{'category':'prompts'}}.\
        User's query: {topics}
    2. Delegate the appropriate prompts to the most suitable agents to work on.
    3. Craft your response based on the output produced by the agents who have been delegated work only.
    4. Consolidate, analyse and recompile the output from the other agents into a responsed that is \
        logical, coherent, meaningful and targetted to the user's query. 
    5. Craft your response using prose that is succinct and easy to understand, \
        suitable for Post-Secondary School students, typically aged 16 to 17 years old.
    6. Provide links to websites what will be useful to the user in achieving the intended objective(s). 
    """,

    #description="""\ 
    #1. Analyse the work produced by the other agents 
    #2. Consolidate and compile the work provided by the other agents in a logical, coherent and meaningful way. 
    #3. From the output of the previous step, craft your response using ONLY information that is applicable to the user based on his/her query. 
    #4. Craft your response using this information in a manner that is succinct and easy to follow, suitable for Post-Secondary School students.
    #""",

    expected_output="""\
    Very helpful response to the user's query on matters about Post-Secondary Education. \
    """,

    agent=agent_friendly_advisor,

    # tools=[tool_admission_websearch],
)

task_admission = Task(
    description="""\
    1. Understand the user's component prompt(s) in the "Admissions" category.
    2. Research admission matters using the tool provided. 
    3. Respond factually and meaningfully to the user. Where possible, respond only with information that the user is eligible for. 
    4. The timeframe is important. Where applicable, advise on the timeframe of eligible admission exercises.
    5. Include webpage urls as reference where applicable.  
    """,

    expected_output="""\
    Factual and targetted information/advice on matters about Post-Secondary Education admission exercises in response to the user's query.
    """,

    agent=agent_admission,

    tools=[tool_admission_websearch],
)

# Tasks
task_ecg = Task(
    description="""\
    1. Understand the user's component prompt(s) in the "ECG" category.
    2. If possible, perform a Values, Interests, Personality, Skills (VIPS) analysis of the user based on the information \
        provided in the prompt. 
    3. Shortlist not more than 5 Post-Secondary Education only courses/schools suitable for the user based on the VIPS analysis.
    4. Do NOT include University courses.  Respond factually and meaningfully to the user. Where possible, respond only with information that the user is eligible for. 
    5. Include webpage urls as reference where applicable.  
    """,
    
    #description="""\
    #1. Perform Values, Interests, Personality, Skills (V.I.P.S.) analysis of the user based on the information \
    #    provided by the user in his/her query: {topic_ecg}.
    #2. Shortlist not more than 5 Post-Secondary Education only courses/schools suitable for the user based on the analysis.
    #3. Do NOT include University courses.
    #3. Rank the shortlisted courses/schools with the most suitable one first and the least suitable last. 
    #4. Include the url link(s) to the webpage(s) of the identified courses/schools in your list.
    #""",

    expected_output="""\
    Shortlisted Post-Secondary Education only courses/schools that suits the user based on the VIPS analysis. \
        """,

    agent=agent_ecg,

)

# Crew
crew = Crew(
    agents=[agent_friendly_advisor, agent_admission, agent_ecg],
    tasks=[task_craft_response],
    verbose=True
)

# Get current date
from datetime import date
# Kickoff
# Start the crew's task execution
def crew_kickoff(dict_component_queries):
    today = date.today()
    result = crew.kickoff(inputs={
        "topic_admission": f"{dict_component_queries.get('Admission')}",
        "topic_ecg":f"{dict_component_queries.get('ECG')}",
        "topics":f"{json.dumps(dict_component_queries)}. Today's date is {today}"
        })
    return result

def let_the_agents_handle_it(dict_component_queries):
    """if "ECG" in dict_component_queries.keys():
        crew.tasks.append(task_ecg)
    if "Admission" in dict_component_queries.keys():
        # crew.agents.append(agent_admission)
        crew.tasks.append(task_admission)
    if len(crew.tasks)==0:
        response = ""
    else:
        crew.tasks.append(task_craft_response)
        response = crew_kickoff(dict_component_queries).raw"""
    
    # crew.tasks = []
    response = crew_kickoff(dict_component_queries)
    return response
