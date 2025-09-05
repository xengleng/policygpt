#--install packages for this script.
#.venv\Scripts\pip install streamlit
#.venv\Scripts\pip install -qU langchain-deepseek
#.venv\Scripts\pip install -U langchain-huggingface
#.venv\Scripts\pip install -U langchain-chroma
#.venv\Scripts\pip install sentence-transformers
#langchain_community

#-----Import packages-----------------
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

#----- Initialisation -----------------
#set up the embedding model
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#embedding = HuggingFaceEmbeddings(model)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#vector = embeddings.embed_query("Hello world")
#print(vector[:5])  # sample of embedding


#set up chromadb cloud ----------------
chromadb = Chroma(
    collection_name="logistics",
    embedding_function=embeddings,
    chroma_cloud_api_key='ck-9qrfgDi3nVjXcjkMehtYwXNJvq7jBJthmFJviArWsByg',
    tenant='9ada5868-6614-47ce-8d2a-a6b4d099b280',
    database='groundKnowledge',
)

#---- LLM Prompt ---------------------
llm_query = input("How are you today?\n")

print(f"your query: {llm_query}")
#---- RAG from Chromadb -------------
results = chromadb.similarity_search_with_score(
    #"Will it be hot tomorrow?", k=1, filter={"source": "news"}
    llm_query, k=15
)

ranked_results = sorted(results, key=lambda x: x[1])
ranked_results = ranked_results[:3]

#for res, score in ranked_results:
#    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]"

for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

retrieved_doc = [doc.page_content for doc, score in ranked_results]

for i in ranked_results:
  print(i)



#----------- Set up LLM ---------------------

DEEPSEEK_API_KEY = "sk-db9b2cdeb47242caa15565029e49ad58"

llm = ChatDeepSeek(
    api_key = DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#----------- Augmented Prompt Construction ----------------------
prompt = ChatPromptTemplate(
[("system","You are highly professional customer service oriented assistant and your reply has to sound like very helpful to customer.  Use ONLY the provided Retrieved documents to answer but if the retrieved info is truncated then consider only the full sentence. If the answer is not in them, say so. Keep it concise."),
("user",f"""Question: {llm_query}
Retrieved Documents:
{retrieved_doc}
Instructions:
- Answer strictly from the docs above.
- Include inline citations like [Source 1], [Source 2].
- If conflicting, prefer the most specific/recent doc.""")])

system = "You are highly professional and trained customer serivice assistant. You are precise in your response. You will also not react to any unethical, wrongdoings or profanity. Use ONLY the provided Retrieved documents to answer. If the answer isn't in them, say so. Keep it concise."
# Initialize conversation history if it doesn't exist
if 'conversation_history' not in globals():
      conversation_history = [{"role": "system", "content": system}]

# Add the user's message to the history
conversation_history.append({"role": "user", "content": llm_query})

#----------- Call LLM ---------------------------------
response = llm.invoke(prompt.format_prompt(llm_query=llm_query, retrieved_doc=retrieved_doc).to_messages())
message_content = response.content

# Add the assistant's reply to the history
conversation_history.append({"role": "assistant", "content": message_content})

print(response.content)

#----------------------------------------------------