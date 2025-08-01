import tempfile
import os
from typing import List, Dict, Optional, Any
from io import BytesIO
import logging

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredURLLoader, 
    TextLoader, 
    PyPDFLoader, 
    UnstructuredFileLoader
)
from langchain_community.vectorstores import Chroma
from fastapi import UploadFile

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Handles retrieval-augmented generation pipeline"""
    
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1024,
            chunk_overlap=300
        )
        self.document_count = 0
        logger.info("RAG Pipeline initialized")
    
    def _split_and_store_documents(self, documents: List[Any], embeddings: Any) -> int:
        """Split documents and store in vector database"""
        if not documents:
            return 0
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
        
        # Initialize or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            logger.info("Created new vector store")
        else:
            self.vectorstore.add_documents(documents=splits)
            logger.info("Added documents to existing vector store")
        
        self.document_count += len(documents)
        return len(documents)
    
    def add_files(self, files: List[UploadFile], embeddings: Any) -> int:
        """Process uploaded files and add to vector store"""
        documents = []
        processed_count = 0
        
        for i, file in enumerate(files, start=1):
            try:
                # Get file extension
                file_name = file.filename or f"file_{i}"
                file_type = file_name.split('.')[-1].lower() if '.' in file_name else ''
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                    content = file.file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Load document based on file type
                    if file_type == "txt":
                        loader = TextLoader(temp_file_path)
                        file_docs = loader.load()
                        
                        for doc in file_docs:
                            doc.metadata['source'] = f'File {i} ({file_name})'
                    
                    elif file_type == "pdf":
                        loader = PyPDFLoader(temp_file_path)
                        file_docs = loader.load()
                        
                        for j, page in enumerate(file_docs, start=1):
                            page.metadata['source'] = f'File {i} ({file_name}) - page {j}'
                    
                    else:
                        # Try generic unstructured loader
                        loader = UnstructuredFileLoader(temp_file_path)
                        file_docs = loader.load()
                        
                        for doc in file_docs:
                            doc.metadata['source'] = f'File {i} ({file_name})'
                    
                    documents.extend(file_docs)
                    processed_count += 1
                    logger.info(f"Successfully processed file: {file_name}")
                
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                continue
        
        # Store documents in vector database
        if documents:
            self._split_and_store_documents(documents, embeddings)
        
        logger.info(f"Processed {processed_count}/{len(files)} files successfully")
        return processed_count
    
    def add_urls(self, urls: List[str], embeddings: Any) -> int:
        """Process URLs and add to vector store"""
        try:
            # Filter out empty URLs
            valid_urls = [url.strip() for url in urls if url.strip()]
            if not valid_urls:
                return 0
            
            # Load documents from URLs
            loader = UnstructuredURLLoader(urls=valid_urls)
            documents = loader.load()
            
            # Add URL metadata
            for i, doc in enumerate(documents):
                if i < len(valid_urls):
                    doc.metadata['source'] = f'URL: {valid_urls[i]}'
            
            # Store documents
            processed_count = self._split_and_store_documents(documents, embeddings)
            logger.info(f"Successfully processed {processed_count} URLs")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing URLs: {e}")
            return 0
    
    def query(self, query: str, llm: Any) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        if self.vectorstore is None:
            raise ValueError("No documents in vector store. Please upload documents first.")
        
        try:
            # Create retrieval chain
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, 
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            
            # Run query
            response = chain.invoke({"question": query}, return_only_outputs=True)
            
            answer = response.get('answer', 'No answer provided')
            sources = response.get('sources', 'No sources provided')
            
            # Parse sources into list if they're comma-separated
            if isinstance(sources, str) and sources != 'No sources provided':
                sources = [s.strip() for s in sources.split(',') if s.strip()]
            elif sources == 'No sources provided':
                sources = []
            
            logger.info(f"RAG query processed successfully")
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise
    
    def has_documents(self) -> bool:
        """Check if the pipeline has any documents"""
        return self.vectorstore is not None and self.document_count > 0
    
    def get_document_count(self) -> int:
        """Get the number of documents in the pipeline"""
        return self.document_count
    
    def clear_documents(self):
        """Clear all documents from the pipeline"""
        self.vectorstore = None
        self.document_count = 0
        logger.info("All documents cleared from RAG pipeline")
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get similar documents for a query (useful for debugging)"""
        if self.vectorstore is None:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {e}")
            return []