# Common imports
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
import streamlit as st
# import lolviz

# Import the key CrewAI classes
from crewai import Agent, Task, Crew, LLM

#if load_dotenv('.env'):
#    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
#else:
#    OPENAI_KEY = st.secrets['OPENAI_API_KEY']

# client = OpenAI(api_key=OPENAI_KEY)

llm=LLM(model=os.getenv('OPENAI_MODEL_NAME'), temperature=0.1)

from pathlib import Path
print(Path(__file__).resolve().parent)

# from crewai_tools import PDFSearchTool
# tool_courses_offered_pdfsearch = PDFSearchTool(pdf="C:\\Users\\Ginno YONG\\Documents\\GovTech AI Champions\\POC\\JAE_Courses.pdf")

# Create a new instance of the WebsiteSearchTool
# Set the base URL of a website, e.g., "https://example.com/", so that the tool can search for sub-pages on that website

from crewai_tools import WebsiteSearchTool
tool_admission_websearch = WebsiteSearchTool("https://www.moe.gov.sg/post-secondary/admissions")
tool_moe_coursefinder_websearch = WebsiteSearchTool("https://www.moe.gov.sg/coursefinder")
tool_np_courses_websearch = WebsiteSearchTool("https://www.np.edu.sg/schools-courses")
tool_sp_courses_websearch = WebsiteSearchTool("https://www.sp.edu.sg/sp/education")
tool_tp_courses_websearch = WebsiteSearchTool("https://www.tp.edu.sg/schools-and-courses/students.html")
tool_nyp_courses_websearch = WebsiteSearchTool("https://www.nyp.edu.sg/student/study/find-course")
tool_rp_courses_websearch = WebsiteSearchTool("https://www.rp.edu.sg/schools-courses/courses")
tool_ite_courses_websearch = WebsiteSearchTool("https://www.ite.edu.sg/courses/course-finder")
#tool_job_roles_websearch = WebsiteSearchTool("https://www.myskillsfuture.gov.sg/content/student/en/secondary/assessment/matching-job-roles.html")

from crewai_tools import SerperDevTool
search_tool = SerperDevTool(country='sg')
search_tool_moe_coursefinder = SerperDevTool(search_url='https://www.moe.gov.sg/coursefinder')
search_tool_np = SerperDevTool(search_url="https://www.np.edu.sg/schools-courses")
search_tool_sp = SerperDevTool(search_url="https://www.sp.edu.sg/sp/education")
search_tool_tp = SerperDevTool(search_url="https://www.tp.edu.sg/landing/students.html")
search_tool_nyp = SerperDevTool(search_url="https://www.nyp.edu.sg/student/study/find-course")
search_tool_rp = SerperDevTool(search_url="https://www.rp.edu.sg/schools-courses/courses")
search_tool_ite = SerperDevTool(search_url="https://www.ite.edu.sg/courses/course-finder")

# Agents
agent_course_finder = Agent(
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
        For example, a course has a reference score range of 10 to 15. \
        A student with a score of <10 will have very good chance of getting into this course, \
        while a student with a score of between 10 and 15 will have a good chance of getting into the same course. \
        A student with a score of >15 will have low chance. 
        In summary, the lower the student's score, the better is his/her chances of getting into a course. 
    """,


    #tools=[tool_np_courses_websearch, tool_sp_courses_websearch, tool_tp_courses_websearch, 
    #     tool_nyp_courses_websearch, tool_rp_courses_websearch, tool_ite_courses_websearch,tool_moe_coursefinder_websearch],

    tools=[search_tool_np, search_tool_sp, search_tool_tp, 
         search_tool_nyp, search_tool_rp, search_tool_ite,search_tool_moe_coursefinder],

    # tools=[search_tool],

    llm=llm,

)

agent_admission = Agent(
    role="Admission Expert",

    goal="Provide clear, factual and targetted information and advice on post secondary school education admission criteria and procedure \
        specific to the user's query",

    backstory="""You are an expert in Singapore's Minstry of Education Post Secondary School admission matters.
            You will advise factually and truthfully the user's query about Post Secondary School admission
            """,

    allow_delegation=False, # we will explain more about this later

	verbose=True, # to allow the agent to print out the steps it is taking

    llm=llm,
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

    # tools=[tool_np_courses_websearch, tool_sp_courses_websearch, tool_tp_courses_websearch, 
    #       tool_nyp_courses_websearch, tool_rp_courses_websearch, tool_ite_courses_websearch],

    llm=llm,

)

# Tasks
task_course_finder = Task(
    description="""\
    1. Provide factual answer to the user's prompt regarding looking for one or more course or school, delimited in <prompt_course_finder> tags. 
    2. <prompt_course_finder>{topic_course_finder}</prompt_course_finder>
    3. Look across all the schools and institutes provided in your tools to generate your answer. Do not include universities.
    4. List the actual courses.
    5. Do NOT hallucinate. Do NOT make up courses that do not exist. Just state so if there are no courses that answers the user's prompt.
    6. Be specific and factual in your response. 
    7. Select only courses where the user has a good or better chance of being accepted.
    8. If the user is not eligible for any course, just state so.  
    9. If the information provided in the user's query is insufficient for you to identify specific courses, \
    suggest how the user may improve his/her query.
    10. Always indicate the type of Aggregate Score, such as ELR2B2-A, ELR2B2-B, etc, or the qualifications and subject passes required where possible.
    11. Include the url link to the exact course webpages of the courses identified in your response.
    """,
    
    expected_output="""\
    Factual information about Post-Secondary School courses targetted to answer the user's prompt: {topic_course_finder}.\
        """,

    agent=agent_course_finder,

    # tools=[search_tool, search_tool_moe_coursefinder],
    #tools=[tool_moe_coursefinder_websearch, tool_np_courses_websearch, tool_sp_courses_websearch, 
    #       tool_tp_courses_websearch, tool_nyp_courses_websearch, tool_rp_courses_websearch, tool_ite_courses_websearch],

)

task_course_info = Task(
    description="""\
    1. Find out about the course(s) as indicated in the user's prompt delimited in <prompt_course_info> tags. 
    2. <prompt_course_info>:{topic_course_info}</prompt_course_info>
    3. Be specific and factual in your response, do NOT make up information. 
    4. List the actual courses. Do NOT make up courses. 
    4. Include the url links to the exact webpages of courses that form the source of your information. 
    """,
    
    expected_output="""\
    Factual and targetted information about a school or course that answers the user's prompt: {topic_course_info} \
        """,

    agent=agent_course_info,

    context=[task_course_finder],

    #tools=[tool_moe_coursefinder_websearch, search_tool],
    tools=[tool_moe_coursefinder_websearch, search_tool],
    #tools=[tool_moe_coursefinder_websearch, search_tool_np, search_tool_sp, search_tool_tp, search_tool_nyp, search_tool_rp, search_tool_ite],
)

task_admission = Task(
    description="""\
    1. Provide targetted answer to the user's prompt on Post Secondary School Education admission, delimited in <prompt_admission> tags.
    2. <prompt_admission>:{topic_admission}</prompt_admission>
    3. Be specific and factual in your response. Do NOT make up answers.
    4. For each Post-Secondary School Education admission exercise, indicate if the user is eligible for it \
        if you have enough information to determine. 
    5. Include the url links to the exact webpages of the source of your information as reference. 
    6. Be mindful of the timeframe of each admission exercise and the date of the user's prompt when generating your answer. 
    """,

    expected_output="""\
    Factual and targetted information about Post-Secondary Education admission exercises that answers the user's prompt: {topic_admission}.
    """,

    agent=agent_admission,

    tools=[tool_admission_websearch],

    context=[task_course_finder, task_course_info]
)

# Crew
crew = Crew(
    #agents=[agent_admission, agent_ecg, agent_course_finder, agent_course_info],
    agents=[agent_course_finder, agent_course_info, agent_admission],
    tasks=[task_course_finder, task_course_info, task_admission],
    verbose=True,
)

# Get current date
from datetime import date

# Kickoff
# Start the crew's task execution
def crew_kickoff(dict_component_queries):
    topic_admission = None
    if dict_component_queries.get('Admission') is not None:
        topic_admission = dict_component_queries.get('Admission') + "; Date today:"+ date.today().strftime('%d/%m/%Y')

    result = crew.kickoff(inputs={
        "topic_admission": f"{topic_admission};",
        "topic_course_finder":f"{dict_component_queries.get('CourseFinder')}",
        "topic_course_info":f"{dict_component_queries.get('AboutCourse')}",
        #"topics":f"{json.dumps(dict_component_queries)}",
        #"topics":f"{user_query}. Today's date is {today}",
        })
    return result

def let_the_agents_handle_it(dict_component_queries):
    crew.tasks = []
    
    #if "ECG" in dict_component_queries.keys():
    #    crew.tasks.append(task_ecg)

    if "CourseFinder" in dict_component_queries.keys():
        crew.tasks.append(task_course_finder)

    if "AboutCourse" in dict_component_queries.keys():
        crew.tasks.append(task_course_info)

    if "Admission" in dict_component_queries.keys():
        crew.tasks.append(task_admission)

    
    # crew.tasks.append(task_craft_response)
    
    # response = crew_kickoff(dict_component_queries).raw

    #if query_type=="CourseFinder":
    #    crew.tasks = [task_course_advisor]
    #else:
    #    crew.tasks = [task_admission]

    
    response = crew_kickoff(dict_component_queries)

    list_task_output=[]
    for i in range(len(response.tasks_output)):
        list_task_output.append(f"""\n\n[Task Ouput {i}]:\n{response.tasks_output[i].raw}""")

    str_task_output = "\n\n".join(list_task_output)
    return str_task_output
