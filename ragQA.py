import pickle, os
from io import BytesIO
import tempfile
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, WebBaseLoader, PyPDFLoader, UnstructuredFileLoader, AzureAIDocumentIntelligenceLoader
from langchain_community.vectorstores import FAISS, Chroma
import panel as pn
import bs4
pn.extension()

def split_and_store_vectorDB(data, vectorstore, embeddings): 
    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=2048,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(data)
    # print(len(splits))

    # Vector Database
    if vectorstore is None:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    else:
        vectorstore.add_documents(documents=splits, embedding=embeddings)

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

    return vectorstore

def prepare_vectorDB_files(file_path_list, vectorstore, embeddings): 
    # Load data
    data = []
    for i, file_path in enumerate(file_path_list, start=1): 
        print(i, file_path)
        file_name = file_path.split("/")[-1]
        file_type = file_path.split('.')[-1].lower() if '.' in file_path else ''
        print(file_name)
        print(file_type)

        if file_type=="txt": 
            loader = TextLoader(file_path)
            one_file = loader.load()

            for doc in one_file:
                doc.metadata['source'] = f'File {i} ({file_name})'

        elif file_type=="pdf": 
            loader = PyPDFLoader(file_path)
            one_file = loader.load()
            for j, page in enumerate(one_file, start=1):
                page.metadata['source'] = f'File {i} ({file_name}) - page {j}'

            """
        elif file_type=="docx": 
            loader = AzureAIDocumentIntelligenceLoader(file_path=file_path, api_model="prebuilt-layout")
            one_file = loader.load()
            """
        else: 
            raise TypeError(f"We don't allow the file type {file_type}. ")

        data.extend(one_file)
    
    vectorstore = split_and_store_vectorDB(data, vectorstore, embeddings)
    return vectorstore

def prepare_vectorDB_URLs(URL_inputs, vectorstore, embeddings): 
    # print([url.value for url in URL_inputs])
    loaders = UnstructuredURLLoader(urls=[url.value for url in URL_inputs])
    data = loaders.load() 

    vectorstore = split_and_store_vectorDB(data, vectorstore, embeddings)
    return vectorstore

def ragQA(question, user, instance, vectorstore, llm):
    
    # Retrieval
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}))
    # basic_rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
    response = chain.invoke({"question": question}, return_only_outputs=True)
    # response = basic_rag_chain.invoke(query)

    print(response)
    answer = response.get('answer', 'No answer provided')
    sources = response.get('sources', 'No sources provided')
    return {'object': answer + f"\n The sources are: {sources}"}
