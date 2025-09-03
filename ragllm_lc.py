from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_chroma import Chroma
import requests
import streamlit as st

chromadb = Chroma(
    collection_name="logistics",
    embedding_function=embedding,
    chroma_cloud_api_key=st.secrets["CHROMADB_API_KEY"],
    tenant=st.secrets["CHROMADB_TENANT"],
    database='groundKnowledge',
)

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    api_key = st.secrets["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# ---- Enhanced Chat Response Function ----
def generate_response(prompt, chat_history):
    """
    Generate a response based on user input and chat history
    This simulates a more intelligent response that considers conversation context
    """
    st.session_state.processing = True
    
    # --- RAG semantic search from chromaDB  ---

    results = chromadb.similarity_search_with_score([prompt], k=15)
    ranked_results = sorted(results, key=lambda x: x[1])
    ranked_results = ranked_results[:3]
    
    retrieved_doc = [doc.page_content for doc, score in ranked_results]

 # --- Create Augmented Prompt ---
    from langchain_core.prompts import ChatPromptTemplate

    system = "You are highly professional customer service oriented assistant and your reply has to sound like very helpful to customer.  Use ONLY the provided Retrieved documents to answer but if the retrieved info is truncated then consider only the full sentence. If the answer is not in them, say so. Keep it concise."

    prompt = ChatPromptTemplate(
    [("system", {system}),
    ("user",f"""Question: {prompt}
    Retrieved Documents:
    {retrieved_doc}
    Instructions:
    - Answer strictly from the docs above.
    - Include inline citations like [Source 1], [Source 2].
    - If conflicting, prefer the most specific/recent doc.""")])


    # Initialize conversation history if it doesn't exist
    if 'conversation_history' not in globals():
        conversation_history = [{"role": "system", "content": system}]

    user = f"""Question: {prompt}
    Retrieved Documents:
    {retrieved_doc}
    Instructions:
    - Answer strictly from the docs above.
    - Include inline citations like [DOC 2], [DOC 4].
    - If conflicting, prefer the most specific/recent doc."""

    # Add the user's message to the history
    conversation_history.append({"role": "user", "content": user})
    response = llm.invoke(prompt.format_prompt(llm_query=llm_query, retrieved_doc=retrieved_doc).to_messages())
    message_content = response.content


    # Add the assistant's reply to the history
    conversation_history.append({"role": "assistant", "content": message_content})

    # Print the message content. If the content contains markdown,
    # Colab will render it properly when printed.
        
    st.session_state.processing = False
    return message_content

