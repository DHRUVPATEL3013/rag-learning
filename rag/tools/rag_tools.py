from typing import TypedDict,Annotated,List
from langchain_core.messages import BaseMessage,HumanMessage,ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
# from retriever.dense_retriever import dense_retriver
from rag.retriever.retriever import final_retriver
from rag.llm import llm


class AgentState(TypedDict):
  messages:Annotated[List[BaseMessage],add_messages]


@tool
def retrieve_documents(query:str) ->str:
  """Search the document for releavant information for the question.
  use for question which need information from the documents.
  """
  docs=final_retriver.invoke(query)
  return "\n\n".join(d.page_content for d in docs)

@tool
def calculator(expression:str)->str:
  """Evaluate a mathematical expression
  use for arithmatic or mathematical calculations.
  
  """
  try:
    return eval(expression)
  except Exception as e:
    return f"Error: {e}"

tools=[retrieve_documents,calculator]

llm_with_tools=llm.bind_tools(tools)



def agent_node(state:AgentState):
  response=llm_with_tools.invoke(state["messages"])
  return {"messages":[response]}

def tool_node(state:AgentState):
   last_message=state["messages"][-1]
   results=[]
   for tool_call in last_message.tool_calls:
      for t in tools:
         if t.name == tool_call["name"]:
            output=t.invoke(tool_call["args"])
            results.append(ToolMessage(
               content=str(output),
               tool_call_id=tool_call["id"]
            ))
         break
   return {"messages":results}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
# graph.add_node("tools", ToolNode(tools))   # auto-executes tools
graph.add_node("tools",tool_node)
graph.set_entry_point("agent")

graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END,
})

graph.add_edge("tools", "agent")

app = graph.compile()


# ----- Test -----
if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="What is 25*4?")]
    })
    print(result["messages"][-1].content)