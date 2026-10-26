from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#using only the context below.
#if the answer not in the context.say "I don't know" 

prompt=ChatPromptTemplate.from_template(
  """
  Answer the question from the provided context only.if you dont get the answer please reply with "I Don't Know"
  Context:
  {context}
  
  Question:{question}
  
  Answer:
  """
)

relevant_prompt = ChatPromptTemplate.from_template(
    """You are an evaluation assistant.

Your job: check if the answer actually addresses the question.

**YES** — The answer directly addresses what was asked.
- Examples: Question asks "Who invented X?" and answer names a person.
- Question asks "What are the types of X?" and answer lists types.

**NO** — The answer does not address the question.
- Example: Question asks "Who invented X?" and answer describes what X is.
- Example: Question asks "What are the types?" and answer explains history.
- Example: Answer is off-topic or unrelated.

Do NOT judge whether the answer is correct or grounded in any context.
Only judge: does the answer address the question?

Question: {question}
Answer: {answer}

Reply with exactly one word: YES or NO."""
)

grounded_prompt = ChatPromptTemplate.from_template(

    """You are an evaluation assistant.

Your job: classify the answer into one of three categories.

**YES** — The answer is fully grounded in the context AND addresses the question.

**NO** — The answer contains claims not supported by the context or something you think that this answer is not from the context or not mentioned in provided context.

**REFUSED** — The answer indicates the information is not available in the context,
regardless of the exact words used.

Context:
{context}

Question: {question}
Answer: {answer}

Reply with exactly one word: YES, NO, or REFUSED."""
)


rewrite_prompt=ChatPromptTemplate.from_template(
  """Rewrite this question to be clearer for a document search.
Keep the meaning same but make it more specific.

Original: {question}

Rewritten:"""
)

hyde_prompt=ChatPromptTemplate.from_template(
"""Write a short paragraph that answers this question.
Do not explain, just answer. Write 3-5 sentences.

Question: {question}

Answer:"""
)



memory_prompt=ChatPromptTemplate.from_messages([
   ("system", """You are a helpful assistant.
   Answer the question using only the context and history.

Rules:
- Answer in 2-3 sentences maximum.
- Only the answer, no explanations.
- No "based on context", no "the text says", no meta commentary.
- If the answer is not in context, say "I don't know."""),

    MessagesPlaceholder(variable_name="history"),  
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]

)

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter for a document search system.

Your task: make the current question fully self-contained.

Reason step by step:
1. Read the conversation history carefully.
2. Identify any words or phrases in the current question that refer to 
   something in the history (pronouns, demonstratives, ordinal references, 
   vague references, omitted subjects).
3. For each reference, find what it points to in the history.
4. Rewrite the question by replacing every reference with the actual content 
   it points to.
5. The rewritten question must be understandable with ZERO conversation context.

Rules:
- Do not add new information.
- Do not answer the question.
- Do not explain your reasoning.
- Return ONLY the rewritten question.

If the question has no references to history, return it unchanged."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Current question: {question}\n\nStandalone question:"),
])


multi_query_prompt=ChatPromptTemplate.from_template(
   """
You are an AI language model assistant.

Your task is to generate 3 different versions of the user's question
so that a vector database can find relevant documents.

Original question:
{question}

Generate 3 alternative questions.
Return only the questions, one per line.
"""
)

grade_prompt = ChatPromptTemplate.from_template(
    """You are a strict classifier. Reply with ONLY ONE WORD: YES or NO.

Does any single one of these documents contain information relevant to answering the question?

Question: {question}
Documents:
{documents}

Reply (YES or NO):"""
)

graph_rewrite_prompt = ChatPromptTemplate.from_template(
    """Rewrite this question to get better search results.
Return only the rewritten question.

Question: {question}
"""
)

decide_prompt = ChatPromptTemplate.from_template(
    """You are a strict binary classification system. 

Your task is to determine whether answering the user's question requires searching external documents/retrieval or if it can be answered using standard general knowledge.

STRICT OUTPUT RULES:
- Output MUST be EXACTLY ONE WORD.
- The ONLY allowed output values are "YES" or "NO".
- Do NOT include any punctuation, quotes, introduction, reasoning, explanations, or extra white space.
- Any output other than "YES" or "NO" is invalid.

CRITERIA:
- Output "YES" if the question asks about specific, private, domain-specific, recent, real-time, or proprietary context that requires external documents.
- Output "NO" if the question can be answered using common sense, basic facts, math, or standard pre-trained knowledge.

EXAMPLES:
Question: What is the capital of France?
Reply: NO

Question: What were our Q3 revenue numbers in the internal database?
Reply: YES

Question: How do plants perform photosynthesis?
Reply: NO

Question: According to the attached company policy, what is the holiday roll-over rule?
Reply: YES

Question: {question}
Reply:"""
)