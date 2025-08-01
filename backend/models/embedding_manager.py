import logging
from typing import Optional

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding model - uses HuggingFace sentence-transformers as default with lazy loading"""
    
    _instance = None
    _embedding = None
    _model_name = "all-MiniLM-L6-v2"
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.embedding = None
            self.model_name = self._model_name
            self._initialized = True
    
    def _initialize_embedding(self):
        """Initialize the default HuggingFace embedding model with multiple fallbacks (lazy loaded)"""
        if self._embedding is not None:
            return self._embedding
            
        logger.info("Initializing embedding model (first time only)...")
        
        # Try 1: langchain_huggingface (preferred)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embedding = HuggingFaceEmbeddings(
                model_name=f"sentence-transformers/{self._model_name}",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"HuggingFace embeddings initialized: {self._model_name}")
            return self._embedding
        except Exception as e:
            logger.warning(f"langchain_huggingface not available: {e}")
        
        # Try 2: sentence-transformers with custom wrapper
        try:
            from sentence_transformers import SentenceTransformer
            from langchain.embeddings.base import Embeddings
            
            class SentenceTransformerEmbeddings(Embeddings):
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()

                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
            
            self._embedding = SentenceTransformerEmbeddings(f"{self._model_name}")
            logger.info(f"Sentence-transformers embeddings initialized: {self._model_name}")
            return self._embedding
        except Exception as e:
            logger.warning(f"sentence-transformers not available: {e}")
        
        # Try 3: OpenAI embeddings (if API key available)
        try:
            import os
            if os.getenv('OPENAI_API_KEY'):
                from langchain_openai import OpenAIEmbeddings
                self._embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
                self._model_name = "OpenAI text-embedding-ada-002"
                logger.info("Fallback to OpenAI embeddings (API key found)")
                return self._embedding
        except Exception as e:
            logger.warning(f"OpenAI embeddings not available: {e}")
        
        # Try 4: Dummy embeddings for testing (last resort)
        try:
            from langchain.embeddings.base import Embeddings
            import numpy as np
            
            class DummyEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    return [np.random.rand(384).tolist() for _ in texts]

                def embed_query(self, text):
                    return np.random.rand(384).tolist()
            
            self._embedding = DummyEmbeddings()
            self._model_name = "dummy-embeddings"
            logger.warning("Using dummy embeddings - install sentence-transformers for proper functionality")
            return self._embedding
        except Exception as e:
            logger.error(f"Even dummy embeddings failed: {e}")
        
        raise RuntimeError("Could not initialize any embedding model. Please install sentence-transformers: pip install sentence-transformers")
    
    def get_embeddings(self) -> any:
        """Get the embedding instance (lazy loaded)"""
        if self._embedding is None:
            self._initialize_embedding()
        return self._embedding
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        return {
            "name": f"sentence-transformers/{self._model_name}",
            "type": "HuggingFace Sentence Transformer",
            "available": True,
            "description": "Lightweight, fast, and effective sentence embeddings (lazy loaded)"
        }