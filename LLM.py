import os
import panel as pn
from dotenv import load_dotenv

load_dotenv()

def load_LLM(llm_name): 
    if llm_name == "GPT-3.5 Turbo": 
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv('OPENAI_API_KEY'))
    elif llm_name == "Meta-Llama-3.1-8B": 
        from langchain_groq import ChatGroq
        llm = ChatGroq(temperature=0.6, model_name='llama-3.1-8b-instant', groq_api_key=os.getenv('GROQ_API_KEY'))
    else: 
        raise ValueError("LLM choice is not provided. ")

    return llm


def load_embeddings(embedding_name): 
    if embedding_name == "OpenAI Embeddings": 
        from langchain_openai import OpenAIEmbeddings
        # os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    elif embedding_name == "Google GenerativeAI Embeddings": 
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=os.getenv('GOOGLE_API_KEY'))
    else: 
        raise ValueError("Embedding choice is not provided. ")

    return embeddings


def run_llm(userMessage, user, chat_interface, llm):
    messages = [{"role": "user", "content": userMessage}]
    
    chat_response = llm.invoke(input = messages)

    return chat_response.content
