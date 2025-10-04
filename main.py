#my first AI agent

from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic_core.core_schema import model_field
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI


load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task, desc=None):
    """Add a new task to the user´s task list. use this when the user wants to add or create a new task"""
    todoist.add_task(content=task,
                     description=desc)

@tool
def show_tasks():
    """Show all tasks from Todoist. use this tool when user wants to see their tasks"""
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks



    return tasks
tools = [add_task, show_tasks]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key= gemini_api_key,
    temperature=0.3
    )

system_prompt = """You are a helpful assistant. 
                You will help the user add tasks
                You will help the user show existing tasks. for example , "show me the tasks
                print out the tasks to the user. print them in bullet points
                """



prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),

])


#chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)

agent_executer = AgentExecutor(agent= agent, tools=tools, verbose=True)

#response = chain.invoke({"input": user_input})



history = []
while True:
    user_input = input("Your input:")
    response = agent_executer.invoke({"input": user_input, "history": history})
    print(response["output"])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response["output"]))

