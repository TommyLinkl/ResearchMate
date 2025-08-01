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
rag_pipeline = RAGPipeline()
metrics_logger = MetricsLogger()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    llm_model: Optional[str] = "GPT-3.5 Turbo"
    use_retrieval: bool = True
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    answer: str
    sources: Optional[List[str]] = None
    model_used: str
    response_time: float
    session_id: str
    timestamp: datetime

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
        
        if request.use_retrieval and rag_pipeline.has_documents():
            # Use RAG pipeline
            result = rag_pipeline.query(request.query, llm)
            answer = result.get('answer', 'No answer provided')
            sources = result.get('sources', [])
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
            timestamp=datetime.now()
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
        
        # Process files
        processed_count = rag_pipeline.add_files(files, embeddings)
        
        return {
            "message": f"Successfully processed {processed_count} files",
            "files_processed": processed_count,
            "total_documents": rag_pipeline.get_document_count()
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
        
        # Process URLs
        processed_count = rag_pipeline.add_urls(urls, embeddings)
        
        return {
            "message": f"Successfully processed {processed_count} URLs",
            "urls_processed": processed_count,
            "total_documents": rag_pipeline.get_document_count()
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
        "total_documents": rag_pipeline.get_document_count(),
        "active_model": llm_manager.get_active_model(),
        "embedding_model": embedding_manager.get_model_info()["name"],
        "average_response_time": metrics_logger.get_average_response_time()
    }

@app.delete("/api/documents")
async def clear_documents():
    """Clear all uploaded documents"""
    try:
        rag_pipeline.clear_documents()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)