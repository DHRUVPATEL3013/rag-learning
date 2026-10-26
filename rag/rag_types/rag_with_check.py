import pickle
from langchain_core.output_parsers import StrOutputParser
from rag.llm import llm, format_docs
from rag.prompts.prompts import prompt
from rag.config.config import parent_store
from rag.multi_check import is_safe,is_grounded,is_relevance
import time
from rag.logging_response.structured_logging import log_query
from rag.context.llm_context import reorder_docs,fit_to_budget


rag_chain=prompt | llm | StrOutputParser()


def rag_with_check(question,retriever):

    start= time.time()
    docs=retriever.invoke(question)
    # parent_ids=set(c.metadata["parent_id"] for c in children_docs)

    # parents=[]
    # for pid in parent_ids:
    #     parent_bytes=parent_store.mget([pid])[0]
    #     parent=pickle.loads(parent_bytes)
    #     parents.append(parent)

    # reordered_docs=reorder_docs(parents)
    # final_docs=fit_to_budget(reordered_docs)
    
    context = format_docs(docs)



    answer=rag_chain.invoke(
    {
        "context":context,
        "question":question
    }
    )

    grounded=is_grounded(context,question,answer)

    relevant=is_relevance(question,answer)

    safe=is_safe(answer)

    latency_ms=int((time.time() - start) * 1000)

    log_query(question,answer,grounded,relevant,latency_ms)

    if grounded and relevant and safe :
        return answer
    return "I don't know"




   

   





