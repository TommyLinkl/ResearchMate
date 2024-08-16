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

def prepare_vectorDB(selected_files, embeddings): 
    # Load data
    data = []
    for i, file_path in enumerate(selected_files, start=1): 
        print(i, file_path)
        file_name = file_path.split("/")[-1]
        file_type = file_path.split('.')[-1].lower() if '.' in file_path else ''
        print(file_name)
        print(file_type)

        if file_type=="txt": 
            loader = TextLoader(file_path)
            one_file = loader.load()

            for doc in one_file:
                doc.metadata['source'] = f'File {i}'

        elif file_type=="pdf": 
            loader = PyPDFLoader(file_path)
            one_file = loader.load_and_split()
            print(one_file)
            print(one_file[0])
            for doc in one_file:
                doc.metadata['source'] = f'File {i}'

        elif file_type=="docx": 
            loader = AzureAIDocumentIntelligenceLoader(api_endpoint=endpoint, api_key=key, file_path=file_path, api_model="prebuilt-layout")


        data.extend(one_file)





    '''
        print(file_input.value[:40])
        if file_input.value:
            # print(vars(file_input))
            file_content = file_input.value
            file_name = file_input.name
            file_type = file_name.split('.')[-1].lower() if '.' in file_name else ''
            print(file_name)
            # print(file_type)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}", mode='wb') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            loader = TextLoader(temp_file_path)
            one_file = loader.load()

            for doc in one_file:
                doc.metadata['source'] = f'File {i}'

            data.extend(one_file)
            os.remove(temp_file_path)
    '''

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
    

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=2048,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(data)
    # print(len(splits))

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
