import os
import pickle
import time
import panel as pn
pn.extension()
pn.config.console_output = 'both'

from ragQA import ragQA
from LLM import load_LLM, load_embeddings, run_llm

######################################################
file_input = pn.widgets.FileInput(accept=".txt,.docx,.pdf,.dat", height=50)

llm_options = ["GPT-3.5 Turbo", "Meta-Llama-3.1-8B"]
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic("model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024")

# os.environ['COHERE_API_KEY'] = cohere_key
# from langchain_cohere import CohereModel
# llm = CohereModel(model="command-xlarge", temperature=0.7)

# os.environ["GOOGLE_API_KEY"] = getpass.getpass()
# from langchain_google_vertexai import ChatVertexAI
# llm = ChatVertexAI(model="gemini-1.5-flash")

# os.environ["COHERE_API_KEY"] = getpass.getpass()
# from langchain_cohere import ChatCohere
# llm = ChatCohere(model="command-r-plus")

# os.environ["NVIDIA_API_KEY"] = getpass.getpass()
# from langchain import ChatNVIDIA
# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# os.environ["GROQ_API_KEY"] = getpass.getpass()
# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama3-8b-8192")

# os.environ["MISTRAL_API_KEY"] = getpass.getpass()
# from langchain_mistralai import ChatMistralAI
# llm = ChatMistralAI(model="mistral-large-latest")

llm_dropdown = pn.widgets.Select(name='Choose LLM', options=llm_options)

embeddings_options = ["OpenAI Embeddings", "Google GenerativeAI Embeddings"]
# OpenAIEmbeddings(model="text-embedding-ada-002")

# from langchain_cohere import CohereEmbeddings
# embeddings = CohereEmbeddings(model="large")

# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


embeddings_dropdown = pn.widgets.Select(name='Choose embeddings', options=embeddings_options)

def chatBot_dynamic(question, user, instance):
    llm = load_LLM(llm_dropdown.value)
    embeddings = load_embeddings(embeddings_dropdown.value)

    if not file_input.value:
        response = run_llm(question, user, None, llm)
    else:
        response = ragQA(question, user, instance, file_input, llm, embeddings)

    return response

chat_interface = pn.chat.ChatInterface(
    callback=chatBot_dynamic,
    callback_user=f"ChatBot (retrieval mode)" if file_input.value else f"ChatBot"
)
chat_interface.send(
    "Please first choose an LLM and an embedding. \n\nThen you can either \n 1) upload a file and ask specific questions about its content, or \n 2) send a message to get a reply from the LLM without context!", 
    user="System", 
    respond=False
)


sidebar = pn.Column(
    "## Settings",
    llm_dropdown,
    embeddings_dropdown,
    pn.Column("Upload a file for retrieval: \n Allowed formats: .txt, .docx, .pdf, .dat", file_input),
    width=350,  # Fixed width for sidebar
    height=800, 
    sizing_mode="fixed",  # Ensure the sidebar has a fixed width
    margin=(10, 10, 10, 10),  # Add some margin to the sidebar
)
main_content = pn.Column(
    "# Chat with me", 
    chat_interface,
    sizing_mode="stretch_both",  # Allow the main content to stretch to available space
    margin=(10, 10, 10, 10)  # Add margin to ensure content doesn't touch the edges
)
page = pn.Row(
    sidebar,
    main_content,
    sizing_mode="stretch_both"  # Ensure the row stretches to the size of the window
)
page.servable()

