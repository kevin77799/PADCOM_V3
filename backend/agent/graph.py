import os
from typing import TypedDict, List

# Try to import langgraph dependencies - make them optional
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langgraph.graph import StateGraph, END
    import operator
    from typing import Annotated
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"[WARNING] LangGraph components not available: {e}")

# Configuration
LLM_MODEL = "llama3" 
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

# --- 1. Define the State ---
if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        context: str
        language: str

    # --- 2. Initialize Models ---
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- 3. Define Nodes (Actions) ---
    def retrieve(state: AgentState):
        """Searches the Vector Database for relevant personal data/documents."""
        query = state["messages"][-1].content
        
        try:
            if os.path.exists(VECTOR_DB_PATH):
                vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
                results = vectorstore.similarity_search(query, k=3)
                context_text = "\n\n".join([doc.page_content for doc in results])
            else:
                context_text = "No personal data found. (Database not created yet)."
        except Exception as e:
            print(f"Retrieval Error: {e}")
            context_text = ""

        return {"context": context_text}

    def generate(state: AgentState):
        """Generates the final response using the LLM + Context."""
        messages = state["messages"]
        context = state["context"]
        
        system_prompt = (
            f"You are Jarvis, a highly advanced, intelligent, and helpful AI assistant. "
            f"You have access to the user's personal data:\n"
            f"--- DATA START ---\n{context}\n--- DATA END ---\n\n"
            f"Rules:\n"
            f"1. Answer based strictly on the provided data if relevant.\n"
            f"2. If the answer is not in the data, use your general knowledge but mention you are unsure.\n"
            f"3. Be concise, professional, and friendly.\n"
            f"4. Detect the language of the user's last message and respond in the SAME language."
        )
        
        prompt_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(prompt_messages)
        
        return {"messages": [response]}

    # --- 4. Build the Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    agent_workflow = workflow.compile()
else:
    agent_workflow = None