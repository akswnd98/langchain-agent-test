from typing import List, Tuple
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description

def generate_history_passthrough ():
  def give_history_to_chat_prompt_template (inputs):
    intermediate_steps: List[Tuple[AgentAction, str]] = inputs['intermediate_steps']
    ret = []
    for intermediate_step in intermediate_steps:
      ret.append((intermediate_step[0].messages[0].type, f'''Thought: {intermediate_step[0].messages[0].content}'''))
      ret.append(('human', f'Observation: {intermediate_step[1]}'))
    return ret

  return RunnablePassthrough.assign(
    history=give_history_to_chat_prompt_template
  )

from langchain.tools import Tool

def generate_llama3_chat_prompt_template (tools: list[Tool]):
  system_content_prompt = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use only one of following formats:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action

or

Thought: I now know the final answer
Final Answer: the final answer to the original input question'''

  user_content_prompt = f'''Question: {{input}}'''

  return ChatPromptTemplate.from_messages([
    ('system', system_content_prompt),
    ('user', user_content_prompt),
    MessagesPlaceholder(variable_name='history')
  ]).partial(
    tools=render_text_description(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
  )

def generate_gemma2_chat_prompt_template (tools: list[Tool]):
  system_content_prompt = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use only one of following formats:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action

or

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

'''

  user_content_prompt = f'''Question: {{input}}'''

  return ChatPromptTemplate.from_messages([
    # ('system', system_content_prompt),
    ('user', system_content_prompt + user_content_prompt),
    MessagesPlaceholder(variable_name='history')
  ]).partial(
    tools=render_text_description(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
  )

from langchain_community.utilities import SearxSearchWrapper

def generate_tools ():
  searxng_search_wrapper = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

  def searxng_search (*args, **kargs):
    return searxng_search_wrapper.run(*args, **kargs)

  searxng_tool = Tool(
    name='searxng',
    description='Search searxng for recent web results',
    func=searxng_search
  )

  tools = [searxng_tool]

  return tools
