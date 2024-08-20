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


# More LLM's 
'''
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic("model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024")

os.environ['COHERE_API_KEY'] = cohere_key
from langchain_cohere import CohereModel
llm = CohereModel(model="command-xlarge", temperature=0.7)

os.environ["GOOGLE_API_KEY"] = getpass.getpass()
from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(model="gemini-1.5-flash")

os.environ["COHERE_API_KEY"] = getpass.getpass()
from langchain_cohere import ChatCohere
llm = ChatCohere(model="command-r-plus")

os.environ["NVIDIA_API_KEY"] = getpass.getpass()
from langchain import ChatNVIDIA
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

os.environ["GROQ_API_KEY"] = getpass.getpass()
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192")

os.environ["MISTRAL_API_KEY"] = getpass.getpass()
from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(model="mistral-large-latest")
'''







# More embeddings 
'''
OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="large")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

For text data, popular open-source embedding models include Word2Vec, GloVe, FastText or pretrained transformer-based models like BERT or RoBERTa. 
'''