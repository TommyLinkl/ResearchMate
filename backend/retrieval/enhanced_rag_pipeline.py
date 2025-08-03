import tempfile
import os
from typing import List, Dict, Optional, Any, Tuple
from io import BytesIO
import logging
import re
from collections import Counter
import numpy as np

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredURLLoader, 
    TextLoader, 
    PyPDFLoader, 
    UnstructuredFileLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from fastapi import UploadFile

logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    """Advanced RAG pipeline with hybrid search, reranking, and enhanced generation"""
    
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.documents: List[Document] = []
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', '!', '?', ',', ' '],
            chunk_size=1000,  # Slightly smaller for better precision
            chunk_overlap=200,  # Reduced overlap for efficiency
            length_function=len,
        )
        
        self.document_count = 0
        self.query_expansion_enabled = True
        self.reranking_enabled = True
        
        logger.info("Enhanced RAG Pipeline initialized with hybrid search capabilities")
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms for better retrieval"""
        if not self.query_expansion_enabled:
            return [query]
        
        expanded_queries = [query]
        
        # Simple query expansion strategies
        # 1. Add variations with common scientific terms
        science_expansions = {
            "quantum": ["quantum mechanical", "quantum physics", "qubit"],
            "machine learning": ["ML", "artificial intelligence", "AI", "deep learning"],
            "neural network": ["NN", "artificial neural network", "deep network"],
            "algorithm": ["method", "approach", "technique", "procedure"],
            "research": ["study", "investigation", "analysis", "paper"],
            "model": ["framework", "system", "architecture"],
            "data": ["dataset", "information", "observations"],
            "analysis": ["examination", "evaluation", "assessment"]
        }
        
        query_lower = query.lower()
        for term, synonyms in science_expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)
        
        # 2. Add question variations
        if not query.endswith('?'):
            expanded_queries.append(f"What is {query}?")
            expanded_queries.append(f"How does {query} work?")
            expanded_queries.append(f"Explain {query}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        logger.info(f"Expanded query from 1 to {len(unique_queries)} variations")
        return unique_queries[:5]  # Limit to top 5 variations
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_terms: List[str], 
                             corpus_size: int, doc_freq: Dict[str, int]) -> float:
        """Calculate BM25 score for document relevance"""
        k1, b = 1.5, 0.75  # BM25 parameters
        doc_length = len(doc_terms)
        avg_doc_length = np.mean([len(d.page_content.split()) for d in self.documents])
        
        score = 0
        for term in query_terms:
            if term in doc_terms:
                tf = doc_terms.count(term)
                df = doc_freq.get(term, 0)
                idf = np.log((corpus_size - df + 0.5) / (df + 0.5))
                
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                score += idf * tf_component
        
        return score
    
    def _rerank_documents(self, query: str, retrieved_docs: List[Document], 
                         scores: List[float]) -> List[Tuple[Document, float]]:
        """Rerank documents using multiple scoring strategies"""
        if not self.reranking_enabled or not retrieved_docs:
            return list(zip(retrieved_docs, scores))
        
        query_terms = query.lower().split()
        reranked_docs = []
        
        # Calculate document frequencies for BM25
        doc_freq = {}
        for doc in self.documents:
            doc_terms = set(doc.page_content.lower().split())
            for term in doc_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        for i, (doc, original_score) in enumerate(zip(retrieved_docs, scores)):
            # 1. Original retrieval score (normalized)
            retrieval_score = original_score if original_score > 0 else 0.1
            
            # 2. BM25 score
            doc_terms = doc.page_content.lower().split()
            bm25_score = self._calculate_bm25_score(query_terms, doc_terms, len(self.documents), doc_freq)
            
            # 3. Exact match bonus
            exact_match_score = 0
            for term in query_terms:
                if term in doc.page_content.lower():
                    exact_match_score += 1
            exact_match_score = exact_match_score / len(query_terms) if query_terms else 0
            
            # 4. Source quality score (prefer certain sources)
            source_score = 1.0
            source = doc.metadata.get('source', '').lower()
            if 'pdf' in source or 'research' in source or 'paper' in source:
                source_score = 1.2
            elif 'url' in source:
                source_score = 0.9
            
            # 5. Length penalty (prefer moderately sized chunks)
            length_score = 1.0
            doc_length = len(doc.page_content)
            if 200 <= doc_length <= 1500:  # Sweet spot for informative chunks
                length_score = 1.1
            elif doc_length < 100:
                length_score = 0.8
            
            # Combine scores with weights
            combined_score = (
                0.4 * retrieval_score +      # Original semantic similarity
                0.3 * bm25_score +           # Keyword relevance
                0.15 * exact_match_score +   # Exact term matches
                0.1 * source_score +         # Source quality
                0.05 * length_score          # Chunk length appropriateness
            )
            
            reranked_docs.append((doc, combined_score))
        
        # Sort by combined score (descending)
        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranked {len(retrieved_docs)} documents using hybrid scoring")
        return reranked_docs
    
    def _split_and_store_documents(self, documents: List[Any], embeddings: Any) -> int:
        """Split documents and store in both vector and keyword indexes"""
        if not documents:
            return 0
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Filter out empty or very short chunks
        valid_splits = [doc for doc in splits if doc.page_content.strip() and len(doc.page_content.strip()) > 10]
        
        logger.info(f"Split {len(documents)} documents into {len(valid_splits)} valid chunks (filtered from {len(splits)} total)")
        
        if not valid_splits:
            logger.warning("No valid chunks after filtering - documents may be too short or empty")
            return 0
        
        # Store all document chunks for BM25
        self.documents.extend(valid_splits)
        
        # Initialize or update vector store (for semantic search)
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(documents=valid_splits, embedding=embeddings)
            logger.info("Created new vector store")
        else:
            self.vectorstore.add_documents(documents=valid_splits)
            logger.info("Added documents to existing vector store")
        
        # Initialize or update BM25 retriever (for keyword search)
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        
        # Create ensemble retriever combining both approaches
        if self.vectorstore:
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self.bm25_retriever],
                weights=[0.6, 0.4]  # Favor semantic search slightly
            )
            logger.info("Created hybrid ensemble retriever (semantic + keyword)")
        
        self.document_count += len(documents)
        return len(documents)
    
    def add_files(self, files: List[UploadFile], embeddings: Any) -> int:
        """Process uploaded files and add to vector store - same as original"""
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
                            doc.metadata['file_type'] = 'text'
                    
                    elif file_type == "pdf":
                        loader = PyPDFLoader(temp_file_path)
                        file_docs = loader.load()
                        
                        for j, page in enumerate(file_docs, start=1):
                            page.metadata['source'] = f'File {i} ({file_name}) - page {j}'
                            page.metadata['file_type'] = 'pdf'
                    
                    else:
                        # Try generic unstructured loader
                        loader = UnstructuredFileLoader(temp_file_path)
                        file_docs = loader.load()
                        
                        for doc in file_docs:
                            doc.metadata['source'] = f'File {i} ({file_name})'
                            doc.metadata['file_type'] = 'other'
                    
                    documents.extend(file_docs)
                    processed_count += 1
                    logger.info(f"Successfully processed file: {file_name}")
                
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                continue
        
        # Store documents in both vector and keyword indexes
        if documents:
            self._split_and_store_documents(documents, embeddings)
        
        logger.info(f"Processed {processed_count}/{len(files)} files successfully")
        return processed_count
    
    def add_urls(self, urls: List[str], embeddings: Any) -> int:
        """Process URLs and add to vector store - same as original"""
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
                    doc.metadata['file_type'] = 'url'
            
            # Store documents
            processed_count = self._split_and_store_documents(documents, embeddings)
            logger.info(f"Successfully processed {processed_count} URLs")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing URLs: {e}")
            return 0
    
    def _create_enhanced_prompt(self, query: str, context_docs: List[Document]) -> str:
        """Create enhanced prompt with chain-of-thought reasoning"""
        
        # Build context with source attribution
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get('source', f'Document {i}')
            content = doc.page_content.strip()
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with chain-of-thought reasoning
        enhanced_prompt = f"""You are ResearchMate, an advanced AI research assistant. Analyze the provided context and answer the user's question with detailed reasoning.

**User Question:** {query}

**Context Documents:**
{context}

**Instructions:**
1. **Analysis**: First, identify which sources are most relevant to the question
2. **Reasoning**: Explain your thought process step-by-step
3. **Answer**: Provide a comprehensive answer based on the evidence
4. **Citations**: Include specific source references for all claims
5. **Uncertainty**: If information is incomplete or uncertain, acknowledge this

**Response Format:**
Answer: [Your detailed answer here]

Sources: [List the specific sources used, e.g., "Source 1, Source 3"]

**Additional Context:** If relevant, explain any limitations or areas where more information would be helpful.

Remember to:
- Base your answer strictly on the provided context
- Use clear, academic language appropriate for research
- Highlight key insights and connections between sources
- Acknowledge if the question cannot be fully answered with the available information"""

        return enhanced_prompt
    
    def query(self, query: str, llm: Any, use_enhanced_generation: bool = True) -> Dict[str, Any]:
        """Enhanced query processing with hybrid search and reranking"""
        if self.vectorstore is None or not self.documents:
            raise ValueError("No documents in vector store. Please upload documents first.")
        
        try:
            # 1. Query expansion
            expanded_queries = self._expand_query(query)
            logger.info(f"Processing {len(expanded_queries)} query variations")
            
            # 2. Hybrid retrieval for each query variation
            all_retrieved_docs = []
            all_scores = []
            
            for expanded_query in expanded_queries:
                if self.ensemble_retriever:
                    # Use hybrid ensemble retriever
                    docs = self.ensemble_retriever.get_relevant_documents(expanded_query)
                    # For ensemble retriever, we'll use uniform scores and let reranking handle it
                    scores = [1.0] * len(docs)
                else:
                    # Fallback to semantic search only
                    docs_with_scores = self.vectorstore.similarity_search_with_score(expanded_query, k=8)
                    docs = [doc for doc, score in docs_with_scores]
                    scores = [score for doc, score in docs_with_scores]
                
                all_retrieved_docs.extend(docs)
                all_scores.extend(scores)
            
            # Remove duplicates while preserving order and scores
            seen_content = set()
            unique_docs = []
            unique_scores = []
            
            for doc, score in zip(all_retrieved_docs, all_scores):
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
                    unique_scores.append(score)
            
            # 3. Reranking
            reranked_results = self._rerank_documents(query, unique_docs, unique_scores)
            
            # Select top documents after reranking
            top_docs = [doc for doc, score in reranked_results[:6]]  # Top 6 for context
            
            logger.info(f"Selected {len(top_docs)} top documents after hybrid retrieval and reranking")
            
            # 4. Enhanced generation
            if use_enhanced_generation:
                enhanced_prompt = self._create_enhanced_prompt(query, top_docs)
                
                # Use the LLM with enhanced prompt
                response = llm.invoke(enhanced_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # Extract sources from the response or use document sources
                sources = []
                for i, doc in enumerate(top_docs, 1):
                    source = doc.metadata.get('source', f'Document {i}')
                    if f'Source {i}' in answer or source in answer:
                        sources.append(source)
                
                if not sources:  # Fallback: include all sources
                    sources = [doc.metadata.get('source', f'Document {i+1}') 
                              for i, doc in enumerate(top_docs)]
            
            else:
                # Fallback to original RetrievalQAWithSourcesChain
                vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_retriever)
                response = chain.invoke({"question": query}, return_only_outputs=True)
                
                answer = response.get('answer', 'No answer provided')
                sources = response.get('sources', 'No sources provided')
                
                # Parse sources into list if they're comma-separated
                if isinstance(sources, str) and sources != 'No sources provided':
                    sources = [s.strip() for s in sources.split(',') if s.strip()]
                elif sources == 'No sources provided':
                    sources = []
            
            logger.info(f"Enhanced RAG query processed successfully")
            return {
                'answer': answer,
                'sources': sources,
                'retrieved_docs_count': len(unique_docs),
                'reranked': self.reranking_enabled,
                'query_expanded': len(expanded_queries) > 1
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG query: {e}")
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
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        self.document_count = 0
        logger.info("All documents cleared from Enhanced RAG pipeline")
    
    def toggle_query_expansion(self, enabled: bool):
        """Enable or disable query expansion"""
        self.query_expansion_enabled = enabled
        logger.info(f"Query expansion {'enabled' if enabled else 'disabled'}")
    
    def toggle_reranking(self, enabled: bool):
        """Enable or disable document reranking"""
        self.reranking_enabled = enabled
        logger.info(f"Document reranking {'enabled' if enabled else 'disabled'}")
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get similar documents for a query with enhanced scoring"""
        if self.vectorstore is None:
            return []
        
        try:
            # Use hybrid retrieval
            if self.ensemble_retriever:
                docs = self.ensemble_retriever.get_relevant_documents(query)[:k]
                scores = [1.0] * len(docs)  # Ensemble doesn't return scores
            else:
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
                docs = [doc for doc, score in docs_with_scores]
                scores = [score for doc, score in docs_with_scores]
            
            # Apply reranking
            reranked_results = self._rerank_documents(query, docs, scores)
            
            return [
                {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in reranked_results[:k]
            ]
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {e}")
            return []