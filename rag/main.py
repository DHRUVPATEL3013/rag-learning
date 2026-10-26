from rag.chains.rag_chain import rag_chain
from rag.rag_types.rag_with_check import rag_with_check
from rag.rag_types.rag_with_memory import rag_with_memory
from rag.retriever.retriever import final_retriver
from rag.retriever.dense_retriever import dense_retriver


questions = [
"What are the expected working hours for employees? "
]


# dense_retriver_answer = rag_with_check(questions[0],dense_retriver)
# hybrid_retriver_answer = rag_with_check(questions[0],final_retriver)

# print(dense_retriver_answer)
# print(hybrid_retriver_answer)

session_id = "user_321"

# ans1=rag_with_memory(session_id,"what is machine learning?",dense_retriver)
# ans2=rag_with_memory(session_id,"what are its Approaches?",dense_retriver)
ans3=rag_with_memory(session_id,"What are the expected working hours for employees?",dense_retriver)


# print(ans1)
# print("*"*70)
print(ans3)
# print(ans3)





