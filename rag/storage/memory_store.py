import json
import os
from langchain_core.messages import HumanMessage,AIMessage


SESSION_DIR="sessions"

os.makedirs(SESSION_DIR,exist_ok=True)

def save_history(session_id,messages):

  filepath=f"{SESSION_DIR}/{session_id}.json"
  data=[]
  for msg in messages:
    data.append({
      "type":"human" if isinstance(msg,HumanMessage) else "ai",
      "content":msg.content
    })

  with open(filepath,"w") as f:
    json.dump(data,f,indent=2)


def load_history(session_id):
  filepath=f"{SESSION_DIR}/{session_id}.json"

  if not os.path.exists(filepath):
    return []

  with open(filepath) as f:
    data=json.load(f)

  messages=[]

  for item in data:
    if item["type"]=="human":
      messages.append(HumanMessage(content=item["content"]))
    else:
      messages.append(AIMessage(content=item["content"]))
  return messages


  


    



