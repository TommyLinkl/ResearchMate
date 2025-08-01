#!/usr/bin/env python3
"""
ResearchMate Server Runner
Run this script to start the new FastAPI-based ResearchMate server.
"""

import uvicorn
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting ResearchMate FastAPI Server...")
    print("Access the application at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Interactive API at: http://localhost:8000/redoc")
    print("-" * 50)
    
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )