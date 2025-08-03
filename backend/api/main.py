from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import time
from datetime import datetime
import uuid

from ..models.llm_manager import LLMManager
from ..models.embedding_manager import EmbeddingManager
from ..retrieval.rag_pipeline import RAGPipeline
from ..retrieval.enhanced_rag_pipeline import EnhancedRAGPipeline
from ..evaluation.metrics import MetricsLogger

# Configure logging with proper path handling
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'api.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ResearchMate API",
    description="ML-powered research assistant with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with absolute path
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Initialize managers
llm_manager = LLMManager()
embedding_manager = EmbeddingManager()
rag_pipeline = RAGPipeline()  # Original pipeline (fallback)
enhanced_rag_pipeline = EnhancedRAGPipeline()  # Phase 2 enhanced pipeline
metrics_logger = MetricsLogger()

# Configuration flag for enhanced features
USE_ENHANCED_RAG = True  # Can be controlled via environment variable

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    llm_model: Optional[str] = "GPT-3.5 Turbo"
    use_retrieval: bool = True
    session_id: Optional[str] = None
    # Phase 2 Enhanced RAG options
    use_enhanced_rag: Optional[bool] = True
    enable_query_expansion: Optional[bool] = True
    enable_reranking: Optional[bool] = True
    use_chain_of_thought: Optional[bool] = True

class QueryResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    answer: str
    sources: Optional[List[str]] = None
    model_used: str
    response_time: float
    session_id: str
    timestamp: datetime
    # Phase 2 Enhanced RAG metadata
    enhanced_rag_used: Optional[bool] = False
    retrieved_docs_count: Optional[int] = None
    query_expanded: Optional[bool] = False
    reranked: Optional[bool] = False

class ModelInfo(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    name: str
    type: str
    available: bool

class LogEntry(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    timestamp: datetime
    query: str
    response: str
    model_used: str
    response_time: float
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    import os
    # Get the absolute path to the project root
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    frontend_path = os.path.join(current_dir, "frontend", "index.html")
    
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        logger.error(f"Frontend file not found at: {frontend_path}")
        return HTMLResponse(
            content="""
            <html>
                <head><title>ResearchMate - File Not Found</title></head>
                <body>
                    <h1>ResearchMate API is Running!</h1>
                    <p>Frontend file not found. Please check the installation.</p>
                    <p><strong>Expected path:</strong> {}</p>
                    <p><strong>API Documentation:</strong> <a href="/docs">/docs</a></p>
                    <p><strong>Interactive API:</strong> <a href="/redoc">/redoc</a></p>
                </body>
            </html>
            """.format(frontend_path),
            status_code=200
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "ResearchMate API is running",
        "embedding_initialized": embedding_manager.embedding is not None,
        "llm_available": len(llm_manager.llms) > 0,
        "project_root": PROJECT_ROOT
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query with optional RAG retrieval"""
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing query for session {session_id}: {request.query[:100]}...")
        
        # Get the LLM
        llm = llm_manager.get_llm(request.llm_model)
        
        # Initialize response metadata
        enhanced_rag_used = False
        retrieved_docs_count = None
        query_expanded = False
        reranked = False
        
        if request.use_retrieval:
            # Determine which RAG pipeline to use
            use_enhanced = (USE_ENHANCED_RAG and 
                          request.use_enhanced_rag and 
                          enhanced_rag_pipeline.has_documents())
            
            if use_enhanced:
                # Configure enhanced pipeline options
                enhanced_rag_pipeline.toggle_query_expansion(request.enable_query_expansion)
                enhanced_rag_pipeline.toggle_reranking(request.enable_reranking)
                
                # Use Enhanced RAG pipeline (Phase 2)
                result = enhanced_rag_pipeline.query(
                    request.query, 
                    llm, 
                    use_enhanced_generation=request.use_chain_of_thought
                )
                answer = result.get('answer', 'No answer provided')
                sources = result.get('sources', [])
                
                # Extract enhanced metadata
                enhanced_rag_used = True
                retrieved_docs_count = result.get('retrieved_docs_count')
                query_expanded = result.get('query_expanded', False)
                reranked = result.get('reranked', False)
                
                logger.info(f"Enhanced RAG used: docs={retrieved_docs_count}, expanded={query_expanded}, reranked={reranked}")
                
            elif rag_pipeline.has_documents():
                # Fallback to original RAG pipeline
                result = rag_pipeline.query(request.query, llm)
                answer = result.get('answer', 'No answer provided')
                sources = result.get('sources', [])
                
                logger.info("Using original RAG pipeline (fallback)")
            
            else:
                # No documents available
                answer = "I don't have any documents to search through. Please upload some documents first and try again."
                sources = []
        else:
            # Direct LLM query
            answer = llm_manager.run_llm(request.query, llm)
            sources = None
        
        response_time = time.time() - start_time
        
        response = QueryResponse(
            answer=answer,
            sources=sources,
            model_used=request.llm_model,
            response_time=response_time,
            session_id=session_id,
            timestamp=datetime.now(),
            # Enhanced RAG metadata
            enhanced_rag_used=enhanced_rag_used,
            retrieved_docs_count=retrieved_docs_count,
            query_expanded=query_expanded,
            reranked=reranked
        )
        
        # Log the interaction
        metrics_logger.log_query(
            query=request.query,
            response=answer,
            model_used=request.llm_model,
            response_time=response_time,
            session_id=session_id
        )
        
        logger.info(f"Query processed successfully in {response_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents for RAG"""
    try:
        logger.info(f"Uploading {len(files)} files")
        
        # Get embeddings
        embeddings = embedding_manager.get_embeddings()
        
        # Process files in both pipelines (sync them)
        processed_count_original = rag_pipeline.add_files(files, embeddings)
        processed_count_enhanced = enhanced_rag_pipeline.add_files(files, embeddings)
        
        processed_count = max(processed_count_original, processed_count_enhanced)
        
        return {
            "message": f"Successfully processed {processed_count} files",
            "files_processed": processed_count,
            "total_documents": enhanced_rag_pipeline.get_document_count(),
            "enhanced_rag_ready": USE_ENHANCED_RAG
        }
        
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@app.post("/api/upload-urls")
async def upload_urls(urls: List[str] = Form(...)):
    """Process URLs for RAG"""
    try:
        logger.info(f"Processing {len(urls)} URLs")
        
        # Get embeddings
        embeddings = embedding_manager.get_embeddings()
        
        # Process URLs in both pipelines (sync them)
        processed_count_original = rag_pipeline.add_urls(urls, embeddings)
        processed_count_enhanced = enhanced_rag_pipeline.add_urls(urls, embeddings)
        
        processed_count = max(processed_count_original, processed_count_enhanced)
        
        return {
            "message": f"Successfully processed {processed_count} URLs",
            "urls_processed": processed_count,
            "total_documents": enhanced_rag_pipeline.get_document_count(),
            "enhanced_rag_ready": USE_ENHANCED_RAG
        }
        
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing URLs: {str(e)}")

@app.get("/api/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available models"""
    return llm_manager.get_available_models()

@app.post("/api/set-model")
async def set_model(llm_model: str = Form(...)):
    """Set the active LLM model"""
    try:
        llm_manager.set_active_model(llm_model)
        
        return {
            "message": "Model updated successfully",
            "active_llm": llm_model
        }
    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting model: {str(e)}")

@app.get("/api/logs", response_model=List[LogEntry])
async def get_logs(limit: int = 100):
    """Get query logs for evaluation"""
    return metrics_logger.get_logs(limit=limit)

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_queries": metrics_logger.get_query_count(),
        "total_documents": enhanced_rag_pipeline.get_document_count(),
        "active_model": llm_manager.get_active_model(),
        "embedding_model": embedding_manager.get_model_info()["name"],
        "average_response_time": metrics_logger.get_average_response_time(),
        # Phase 2 enhancements
        "enhanced_rag_enabled": USE_ENHANCED_RAG,
        "pipeline_version": "2.0-enhanced" if USE_ENHANCED_RAG else "1.0-basic"
    }

@app.get("/api/enhanced-rag/config")
async def get_enhanced_rag_config():
    """Get current enhanced RAG configuration"""
    return {
        "enhanced_rag_enabled": USE_ENHANCED_RAG,
        "query_expansion_enabled": enhanced_rag_pipeline.query_expansion_enabled,
        "reranking_enabled": enhanced_rag_pipeline.reranking_enabled,
        "features": {
            "hybrid_search": "Semantic + BM25 keyword search",
            "query_expansion": "Automatic query expansion with synonyms and variations",
            "reranking": "Multi-factor document reranking (BM25, exact match, source quality, length)",
            "chain_of_thought": "Enhanced prompting with step-by-step reasoning"
        },
        "pipeline_version": "2.0-enhanced"
    }

@app.post("/api/enhanced-rag/config")
async def update_enhanced_rag_config(
    query_expansion: Optional[bool] = None,
    reranking: Optional[bool] = None
):
    """Update enhanced RAG configuration"""
    try:
        if query_expansion is not None:
            enhanced_rag_pipeline.toggle_query_expansion(query_expansion)
        
        if reranking is not None:
            enhanced_rag_pipeline.toggle_reranking(reranking)
        
        return {
            "message": "Enhanced RAG configuration updated",
            "query_expansion_enabled": enhanced_rag_pipeline.query_expansion_enabled,
            "reranking_enabled": enhanced_rag_pipeline.reranking_enabled
        }
    except Exception as e:
        logger.error(f"Error updating enhanced RAG config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.delete("/api/documents")
async def clear_documents():
    """Clear all uploaded documents"""
    try:
        rag_pipeline.clear_documents()
        enhanced_rag_pipeline.clear_documents()
        return {
            "message": "All documents cleared successfully from both pipelines",
            "pipelines_cleared": ["basic", "enhanced"]
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)