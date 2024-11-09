from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import format_tool_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage, AIMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize tools
tools = [TavilySearchResults(max_results=1)]
tool_executor = ToolExecutor(tools)

# Initialize model with function calling
model = ChatOpenAI(temperature=0, streaming=True)
functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    return "continue"

def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tools(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    function_call = last_message.additional_kwargs["function_call"]
    action = ToolInvocation(
        tool=function_call["name"],
        tool_input=json.loads(function_call["arguments"])
    )
    
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}

# Create and configure workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge('action', 'agent')

app = workflow.compile()

def get_clean_answer(question: str) -> str:
    """
    Get a clean answer from the workflow
    
    Args:
        question (str): The question to ask
        
    Returns:
        str: The final answer, cleaned of any intermediate steps
    """
    inputs = {"messages": [HumanMessage(content=question)]}
    result = app.invoke(inputs)
    
    # Get the last message from the conversation
    messages = result['messages']
    last_message = messages[-1]
    
    # If it's an AI message, return its content
    if isinstance(last_message, AIMessage):
        return last_message.content
    
    # If we didn't get a final AI message, return all available information
    return "Could not get a clean answer. Raw response: " + str(last_message.content)

if __name__ == "__main__":
    question = "what is the weather in frisco, tx"
    answer = get_clean_answer(question)
    print("\nQuestion:", question)
    print("\nAnswer:", answer)