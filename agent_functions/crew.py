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

from pathlib import Path
print(Path(__file__).resolve().parent)

# from crewai_tools import PDFSearchTool
# tool_courses_offered_pdfsearch = PDFSearchTool(pdf="C:\\Users\\Ginno YONG\\Documents\\GovTech AI Champions\\POC\\JAE_Courses.pdf")

# Create a new instance of the WebsiteSearchTool
# Set the base URL of a website, e.g., "https://example.com/", so that the tool can search for sub-pages on that website

from crewai_tools import WebsiteSearchTool
tool_admission_websearch = WebsiteSearchTool("https://www.moe.gov.sg/post-secondary/admissions")
tool_ecg_websearch = WebsiteSearchTool("https://www.moe.gov.sg/coursefinder")
tool_np_courses_websearch = WebsiteSearchTool("https://www.np.edu.sg/schools-courses/full-time-courses")
tool_sp_courses_websearch = WebsiteSearchTool("https://www.sp.edu.sg/sp/education")
tool_tp_courses_websearch = WebsiteSearchTool("https://www.tp.edu.sg/schools-and-courses/students/all-diploma-courses.html")
tool_nyp_courses_websearch = WebsiteSearchTool("https://www.nyp.edu.sg/student/study/find-course")
tool_rp_courses_websearch = WebsiteSearchTool("https://www.rp.edu.sg/schools-courses/courses/full-time-diplomas")
tool_ite_courses_websearch = WebsiteSearchTool("https://www.ite.edu.sg/courses/course-finder")
tool_job_roles_websearch = WebsiteSearchTool("https://www.myskillsfuture.gov.sg/content/student/en/secondary/assessment/matching-job-roles.html")

from crewai_tools import SerperDevTool
search_tool = SerperDevTool(country='sg')
search_tool_moe_coursefinder = SerperDevTool(search_url='https://www.moe.gov.sg/coursefinder')

# Agents
agent_course_advisor = Agent(
    role="Course recommendation and requirement advisor",
    goal="""Provide answers to user's query regarding Post-Secondary School Education courses""",
    backstory="""Using the tools provided, \
        you developed a good understanding of the Post-Secondary School Education courses in Singapore \
        and their entry requirements, such as qualifications and Aggregate Scores, etc.
    Some courses webpage display an Aggregate Score Range, which indicates the highest and lowest scores of students who were accepted into the course this year.
    This score range provides a reference of how likely a student will be successful in getting accepted into the course \
        with his/her own aggregate score.
    About Aggregate Scores: 
        There are different types of aggregate score, such as ELMAB3, ELR2B2, etc.   
        You understand that a lower aggregate score is actually better than a higher score.
        Meaning that a student will have poorer chance of success in getting accepted in a course if his/her aggregate score is higher, \
        especially if it is higher than the bigger value of a course's reference Aggregate Score Range. 
    """,

    #tools=[tool_np_courses_websearch, tool_sp_courses_websearch, tool_tp_courses_websearch, 
    #      tool_nyp_courses_websearch, tool_rp_courses_websearch, tool_ite_courses_websearch],

    tools=[search_tool,search_tool_moe_coursefinder],

)

agent_friendly_advisor = Agent(
    role="Post-Secondary Education Advisor",

    goal="""Work with othe expert agents to craft a targetted and helpful response to user's query on Post-Secondary School Education.\
        The query has been analysed and broken down into categorised prompts in the format of {{'category':'prompts'}}.\
        User's query: {topics}""" ,

    backstory=f"""You are a superb communicator whose field of expertise is in Singapore's Post-Secondary School Education. 
        Your job is to work with other agents in your crew to provide factual, targetted and useful information/advice \
        to the user's query. 
        Be concise, do not include anything that the user is ineligible for or if you are unsure of. 
        Craft your resonse in such a way that is succinct and easy for a Post Secondary School student to understand. 
        The vast majority of Post Secondary School students are between 16 and 17 years old. 
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)

agent_admission = Agent(
    role="Admission Expert",

    goal="Provide clear, factual and targetted information and advice on post secondary school education admission criteria and procedure \
        specific to the user's query",

    backstory="""You are an expert in Singapore's Minstry of Education Post Secondary School admission matters.
    You will provide admission guidance to the user on courses / schools that are recommended by the other agents. 
    If the other agents did not produce any recommendations, \
        you will provide a response that is as specific as possible to the user's query.
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking
)

agent_course_info = Agent(
    role="Course and School Information Expert",

    goal="Provide information about school(s) or course(s) as specified in the user's query. ",

    backstory="""
    You are an expert in Post-Secondary School courses and schools. 
    You know everything about Post-Secondary School courses and schools.
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking

    tools=[tool_np_courses_websearch, tool_sp_courses_websearch, tool_tp_courses_websearch, 
           tool_nyp_courses_websearch, tool_rp_courses_websearch, tool_ite_courses_websearch],
)

agent_course_finder = Agent(
    role="Course and School Finder",

    goal="""Identify the courses / schools that : 
        a) suits the user based on VIPS and other personal factor analysis 
        b) the user is eligible based on his/her qualifications and/or ELR2B2 aggregate score
        """,

    backstory="""
    You are phenomenal in locating Post-Secondary School courses / schools :
    a) that suits the user based on VIPS and other personal factor analysis, and/or
    b) that the user is eligible for based on his/her qualifications and/or ELR2B2 aggregate score.
    """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking

    tools=[tool_np_courses_websearch, tool_sp_courses_websearch, tool_tp_courses_websearch, 
           tool_nyp_courses_websearch, tool_rp_courses_websearch, tool_ite_courses_websearch],
)

agent_ecg = Agent(
    role="Education and Career Guidance Expert",

    goal="Provide guidance to selecting a Post-Secondary School Education courses/schools by analysing the user's \
    Values, Interests, Personality, Skills (VIPS), Strengths, Weaknesses, and Career Aspirations provided in the query.",
    
    backstory="""You are an Education and Career Guidance expert. 
            Your skills are performing VIPS and other factors analysis to provide guidance in selecting \
                suitable Post-Secondary School Education courses/schools. 
            """,

    allow_delegation=True, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking

    tools=[tool_job_roles_websearch],
)

# Tasks
task_course_advisor = Task(
    description="""\
    Answer the user's query: {topics},
    using information that you find in the websites read by the tool(s).
    Look at all the school / institute websites for your answer.   
    List the actual courses.
    Be specific and factual in your response. 
    Select only courses where the user has a good or better chance of being accepted.
    If the user is not eligible for any course, just state so.  
    If the information provided in the user's query is insufficient for you to identify specific courses, \
    suggest how the user may improve his/her query.
    If the user's score is higher than the bigger value of the course's Aggregate Score Range, \
    indicate that the user may still apply for that course though his/her chances of getting accepted may be low.
    For example: for an Aggregate Score Range of 6 to 13, a user will have very good chance of getting accepted if his/her score is less than 6, \
    a good chance of getting accepted if his/her score is between 6 and 13, and poor chance if more than 13.
    In general, the higher the score, the lower the chance of success.  
    Always indicate the type of Aggregate Score, such as ELR2B2-A, ELR2B2-B, etc, or the qualifications and subject passes required where possible.
    Include the url link to the exact course webpages of the courses identified in your response.
    Output in JSON format.  
    """,
    
    expected_output="""\
    Factual information about Post-Secondary School courses targetted to answer the user's query.\
        """,

    agent=agent_course_advisor,

)

task_course_info = Task(
    description="""\
    1. Provide informtion about a school or course.  
    """,
    
    expected_output="""\
    Important information about a school or course \
        """,

    agent=agent_course_info,
)

task_course_finder = Task(
    description="""\
    1. Identify Post-Secondary Education courses / schools that , 
        a) are suitable for the user based on the work of the Education and Career Guidance Expert agent; and/or 
        b) the user is eligible for based on his/her qualifications and/or ELR2B2 aggregate score. \
            Note that the lower the ELR2B2 score, the better. 
    2. Respond with a list of JSON objects only containing important informtion such as: 
    "Institute Name", "School Name", "Course Name", "Course Code", "ELR2B2 Range", etc.\
    Leave blank for any tags that are inapplicable. 
    3. Be extremely factual, do not include anything that does not exist or that the user is ineligible for. 
    """,
    
    expected_output="""\
    A list of JSON object containing information about \
        Post-Secondary Education courses that suits the user and/or that the user is eligible for. \
        """,

    agent=agent_course_finder,
)

task_ecg = Task(
    description="""\
    1. Provide factual and targetted response to the user's query \
        regarding selecting suitable Post-Secondary Education courses / schools. Query: {topics};
    """,
    
    expected_output="""\
    Helpful guidance on what Post-Secondary Education courses that \
         would be suitable to the user. 
        """,

    agent=agent_ecg,

)

task_admission = Task(
    description="""\
    1. Check if the user wants to find out anything about Post Secondary School Education admission in the query: {topics};
    2. If there is none, do not respond anything.
    3. Else, check the output from the Education and Career Guidance Expert agent.
    4. If the Education and Career Guidance Expert agent provided a list of courses / schools in its work, \
    identify the admission exercises that the user should apply for in order to enrol into these courses / schools.
    5. Else, respond directly to the user's query on Post Secondary School Education admission matters.
    5. Highlight the timeframe by comparing today's date with the registration dates of the admission exercises identified. 
    6. Provide the links to the webpages of these identified admission exercises as reference.

    """,

    expected_output="""\
    Factual and targetted information about Post-Secondary Education admission exercises in response a) to the output provided by the Education and Career Guidance Expert agent, \
        or b) to user's query.
    """,

    agent=agent_admission,

    tools=[tool_admission_websearch],

)

task_craft_response = Task(
    description="""\
    1. Understand the user's query, which has been broken down into component prompts and categorised in the format {{'category':'prompts'}}.\
        User's query: {topics}
    2. Make sure the outputs from the other agents work together to answer the user's query.
    3. Rewrite if necessary so that the response is clear and coherent, specific to the user's query, and easy to understand.
    4. Use writing prose suitable for 16 to 17 year old. 
    """

"""
    description='''\
    1. Understand the user's query: {topics}
    2. Delegate the the most suitable agents to work on different parts of the query.
    3. Craft your response based on the output produced by the agents who have been delegated work only.
    4. Combine the work from the other agents coherently, meaningfully and targetted at answering the user's query.
    5. Do not include options that the user is ineligible for. 
    6. Craft your response using prose that is succinct and easy to understand, \
        suitable for Post-Secondary School students, typically aged 16 to 17 years old.
    7. Where applicable, include links to webpages that helps the user achieve the intended objective(s). 
    ''',
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

    context=[task_ecg, task_admission],

)
# Crew
crew = Crew(
    #agents=[agent_admission, agent_ecg, agent_course_finder, agent_course_info],
    agents=[agent_course_advisor, agent_admission],
    tasks=[],
    verbose=True,
)

# Get current date
from datetime import date
# Kickoff
# Start the crew's task execution
def crew_kickoff(user_query):
    today = date.today()
    result = crew.kickoff(inputs={
        #"topic_admission": f"{dict_component_queries.get('Admission')}; Today's date is {today}",
        #"topic_ecg":f"{dict_component_queries.get('ECG')}",
        #"topics":f"{json.dumps(dict_component_queries)}"
        "topics":f"{user_query}. Today's date is {today}"
        })
    return result

def let_the_agents_handle_it(user_query, query_type):
    crew.tasks = []
    
    #if "ECG" in dict_component_queries.keys():
    #    crew.tasks.append(task_ecg)

    #if "CourseFinder" in dict_component_queries.keys():
    #    crew.tasks.append(task_course_finder)

    #if "AboutCourse" in dict_component_queries.keys():
    #    crew.tasks.append(task_course_info)

    #if "Admission" in dict_component_queries.keys():
    #    crew.tasks.append(task_admission)

    
    # crew.tasks.append(task_craft_response)
    
    # response = crew_kickoff(dict_component_queries).raw

    if query_type=="CourseFinder":
        crew.tasks = [task_course_advisor]
    else:
        crew.tasks = [task_admission]

    
    response = crew_kickoff(user_query)
    return response.raw
