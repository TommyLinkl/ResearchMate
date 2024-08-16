import pickle, os
from io import BytesIO
import tempfile
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, WebBaseLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS, Chroma
import panel as pn
import bs4
pn.extension()

def ragQA(question, user, instance, file_input, llm, embeddings):
    # Load data
    file_content = file_input.value
    file_name = file_input.name
    file_type = file_name.split('.')[-1].lower() if '.' in file_name else ''
    # print(file_name)
    # print(file_type)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}", mode='wb') as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    # Determine the appropriate loader based on file type
    loader = TextLoader(temp_file_path)

    '''
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    '''
    # Load and return data
    data = loader.load()
    os.remove(temp_file_path)

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=2048,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(data)
    print(len(splits))

    # Vector Database
    # vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Optionally, store the vector database
    '''
    # Storing vector index create in local
    file_path="vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_openai, f)


    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
    '''

    # Retrieval
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    # basic_rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
    response = chain.invoke({"question": question}, return_only_outputs=True)
    # response = basic_rag_chain.invoke(query)

    answer = response.get('answer', 'No answer provided')
    sources = response.get('sources', 'No sources provided')
    return {'object': answer + f"\n The sources are {sources}"}
