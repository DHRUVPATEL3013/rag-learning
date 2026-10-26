from langchain_core.output_parsers import StrOutputParser
from rag.context.llm_context import fit_to_budget, reorder_docs
from rag.llm import format_docs,llm
from rag.storage.memory_store import save_history,load_history
from langchain_core.messages import HumanMessage,AIMessage
import pickle
from rag.config.config import parent_store
from rag.prompts.prompts import rewrite_prompt,memory_prompt
# memory=ConversationBufferMemory(
#   memory_key="history",
#   return_messages=True
# )


rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def rag_with_memory(session_id,question, retriever):
    # 1. History load
    # history = memory.load_memory_variables({})["history"]
    history=load_history(session_id)

    if history:
        standalone = rewrite_chain.invoke({
            "history": history,
            "question": question,
        }).strip()
        print(standalone)
        print("-------------------")
    else:
        standalone=question
        print(standalone)

    # 2. Retrieve
    docs = retriever.invoke(standalone)

    context = format_docs(docs)

    # 3. Generate with history
    answer = (memory_prompt | llm | StrOutputParser()).invoke({
        "history": history,
        "context": context,
        "question": question,
    })

    # 4. Save to memory
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    save_history(session_id,history)
    # memory.save_context({"input": question}, {"output": answer})

    return answer

