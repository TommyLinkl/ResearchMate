# Detailed New Development Plan

## Project Vision
Position ResearchMate as a comprehensive ML-powered research assistant that demonstrates:
- **LLM fine-tuning** and domain adaptation
- **Advanced RAG** (Retrieval-Augmented Generation) implementation
- **ML model evaluation** and performance optimization
- **Production deployment** and MLOps practices
- **Full-stack integration** of ML models into web applications

---

## Phase 1: Architecture & Infrastructure Setup
**Status**: **COMPLETE**

### Current State Analysis
The existing codebase already implements:
- **FastAPI-based web API** with modern architecture and endpoints
- **Focused LLM support** (3 core models: Llama-3.1-8B, Mistral-7B, ResearchMate-Mistral-7B)
- **Optimized embeddings** (HuggingFace sentence-transformers by default)
- **Enhanced RAG pipeline** using LangChain + ChromaDB/FAISS
- **Document processing** for PDF and text files
- **Comprehensive logging** with SQLite-based metrics

### Completed Infrastructure
1. **Backend API Migration**
   - Migrated to **FastAPI** with comprehensive endpoints:
     - `POST /api/query` - Process user queries with RAG
     - `POST /api/upload` - Handle file uploads  
     - `POST /api/upload-urls` - Process web URLs
     - `GET /api/models` - List available LLM models
     - `POST /api/set-model` - Configure active model
     - `GET /api/logs` - Retrieve query logs
     - `GET /api/stats` - System statistics
     - `DELETE /api/documents` - Clear document database

2. **Logging & Monitoring Infrastructure**
   - SQLite-based structured logging with timestamps, latency, model usage
   - Performance metrics collection and analytics
   - Session-based query tracking
   - Export capabilities (JSON/CSV)

3. **Modular Architecture**
   ```
   ResearchMate/
   ├── backend/
   │   ├── api/          # FastAPI endpoints
   │   ├── models/       # LLM and embedding managers
   │   ├── retrieval/    # RAG pipeline
   │   └── evaluation/   # Metrics and logging
   ├── frontend/         # Modern HTML/JS interface
   ├── logs/            # Application logs
   └── run_server.py    # Server startup script
   ```

### Model Architecture - Simplified & Focused
- **3 Core LLM Models**:
  1. **Llama-3.1-8B** - Primary open-source model (Meta)
  2. **Mistral-7B** - Secondary open-source model (Mistral AI)  
  3. **ResearchMate-Mistral-7B** - Future fine-tuned physics model
- **Single Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (optimal balance of performance and efficiency, with lazy loading for improved startup performance)

---

## Phase 2: Advanced RAG Pipeline Enhancement
**Status**: 🔄 Upgrade Existing Implementation

### Current vs. Target Architecture
**Current**: Simple retrieval → LLM generation
**Target**: Multi-stage pipeline with reranking and verification

### Enhancement Roadmap
1. **Retrieval Improvements**
   - Implement hybrid search (semantic + keyword)
   - Add query expansion and reformulation
   - Multiple embedding strategies (dense + sparse)

2. **Reranking Layer**
   - Cross-encoder reranking for relevance scoring
   - Source quality assessment
   - Temporal relevance weighting

3. **Generation Enhancement**
   - Chain-of-thought prompting for complex queries
   - Few-shot examples for domain-specific formats
   - Citation verification and formatting

4. **Integration Preparation**
   - Prepare architecture for future fine-tuned model integration
   - Implement model selection and fallback logic

---

## Phase 3: Domain-Specific Fine-Tuning (SFT)
**Goal**: Create ResearchMate-Mistral-7B, a physics-specialized model that outperforms general-purpose LLMs on scientific queries.

### Target Model: ResearchMate-Mistral-7B with LoRA
- **Base Model**: Mistral-7B-Instruct-v0.1
- **Fine-tuning Method**: Low-Rank Adaptation (LoRA)
- **Target Domain**: Physics (condensed matter, quantum physics, statistical physics, quantum computing)
- **Integration**: Will replace the placeholder in the LLM manager as the default model

### Dataset Curation Strategy
1. **Public Scientific Datasets**
   - SciQA (Science Question Answering)
   - MathInstruct (Mathematical reasoning)
   - PubMedQA (Biomedical/physics overlap)
   - ArXiv paper abstracts + conclusions

2. **Synthetic Data Generation**
   - Use GPT-4 to generate physics Q&A pairs from research papers
   - Create instruction-following examples with source attribution
   - Generate domain-specific summarization tasks

3. **Data Format**
   ```json
   {
     "instruction": "Explain topological insulators with citations",
     "input": "Context from research papers...",
     "output": "Topological insulators are... [Citation: Paper X, 2023]"
   }
   ```

### Implementation Plan
- Use Hugging Face `transformers` + `peft` for LoRA training
- Track metrics: training loss, validation perplexity, domain-specific accuracy
- Compare performance against base model on physics Q&A benchmarks
- Integrate fine-tuned model into Phase 2 RAG pipeline

---

## Phase 4: Production Deployment & MLOps
**Goal**: Deploy a scalable, monitored ML system demonstrating production best practices.

### Containerization Strategy
```dockerfile
# Multi-stage Docker build
FROM python:3.9-slim as base
# Install dependencies, models
FROM base as api
# FastAPI backend
FROM node:16 as frontend  
# React frontend
FROM nginx as proxy
# Reverse proxy + static serving
```

### Deployment Options
1. **Cloud Platforms**
   - **HuggingFace Spaces**: For demo and community access
   - **Render/Railway**: Simplified deployment
   - **AWS/GCP**: Full production setup with auto-scaling

2. **MLOps Components**
   - **Model versioning**: Track fine-tuned model performance
   - **A/B testing**: Compare model variants in production
   - **Monitoring**: Latency, throughput, error rates
   - **Logging**: Structured logs for debugging and analysis

### API Architecture
```python
# FastAPI backend structure
@app.post("/api/query")
async def process_query(query: QueryRequest):
    # Route to appropriate model based on query type
    # Log request metadata
    # Return structured response with sources
```

---

## Phase 5: Full-Stack Integration & User Experience
**Goal**: Create a polished interface showcasing ML-powered research assistance.

### Frontend Architecture
- **Framework**: React with TypeScript (or lightweight vanilla JS)
- **Key Features**:
  - Real-time query processing with loading states
  - Source highlighting and citation links
  - Model selection interface (fine-tuned vs. commercial models)
  - Query history and bookmarking
  - Performance metrics visualization

### User Experience Flow
1. **Document Upload**: Drag-and-drop interface for papers/URLs
2. **Query Interface**: Smart input with suggestion and auto-completion
3. **Response Display**: Structured answers with expandable source sections
4. **Model Comparison**: Side-by-side comparison of different models
5. **Export Options**: Save results as formatted reports

### Technical Implementation
- **API Integration**: RESTful communication with backend
- **State Management**: Context/Redux for complex UI state
- **Performance**: Lazy loading, caching, and optimization
- **Accessibility**: WCAG compliance and responsive design

---

## Phase 6: Direct Preference Optimization (DPO)
**Goal**: Train the model to prefer accurate, well-sourced answers over hallucinated responses.
**Priority**: ⚠️ Advanced technique - implement after core functionality is stable

### Preference Dataset Creation
Create (prompt, chosen, rejected) triplets where:
- **Chosen**: Accurate answer with proper citations
- **Rejected**: Hallucinated, vague, or incorrect response

### Example Preference Pair
```
Prompt: "Explain quantum entanglement in condensed matter systems"
Chosen: "Quantum entanglement in condensed matter... [Zhang et al., Nature 2023]"
Rejected: "Quantum entanglement is when particles are connected somehow..."
```

### Implementation
- Use TRL (Transformers Reinforcement Learning) library
- Implement DPOTrainer for preference learning
- Focus on citation accuracy and factual consistency

---

## Phase 7: Comprehensive Evaluation Framework
**Goal**: Implement robust metrics for RAG performance and model comparison.
**Priority**: 📊 Lower priority - add after core functionality is working

### Evaluation Metrics
1. **Automated Metrics**
   - **BERTScore**: Semantic similarity between generated and reference answers
   - **ROUGE**: N-gram overlap for summarization tasks
   - **BLEU**: Translation-quality style metrics

2. **RAG-Specific Metrics (using `ragas`)**
   - **Faithfulness**: How grounded the answer is in retrieved context
   - **Answer Relevancy**: How well the answer addresses the question
   - **Context Precision**: Quality of retrieved documents
   - **Context Recall**: Coverage of relevant information

3. **LLM-as-a-Judge Evaluation**
   - Use GPT-4 to score answer quality, citation accuracy, and completeness
   - Domain expert evaluation for physics-specific correctness

### Implementation
- Create evaluation datasets for physics Q&A
- Automated evaluation pipeline with A/B testing capabilities
- Performance dashboards and comparison tools

> **Note**: This phase can be implemented incrementally alongside other phases for continuous improvement, but is not critical for initial deployment.

---

## Implementation Priority & Timeline

### Chronological Development Order
The phases are now organized in the natural development sequence:

| Phase | Focus Area | Complexity | Impact | Status |
|-------|------------|------------|--------|---------|
| **Phase 1** | Architecture & Infrastructure | Medium | High | **COMPLETE** |
| **Phase 2** | RAG Pipeline Enhancement | Low-Medium | High | 🔥 **Next Priority** |
| **Phase 3** | Domain-Specific Fine-Tuning | High | Medium | ⭐ **Planned** |
| **Phase 4** | Production Deployment | Medium | High | ⭐ **Planned** |
| **Phase 5** | Frontend & User Experience | Low-Medium | Medium | **Planned** |
| **Phase 6** | Direct Preference Optimization | High | Low | ⏳ **Optional** |
| **Phase 7** | Comprehensive Evaluation | Medium | Low | ⏳ **Optional** |

### Development Strategy
**Core Development Path (Phases 1-5):**
1. **Phase 1**: Establish solid API foundation and modular architecture - **COMPLETE**
2. **🔥 Phase 2**: Enhance existing RAG capabilities with advanced retrieval and reranking - **NEXT**
3. **Phase 3**: Fine-tune domain-specific model for physics expertise
4. **Phase 4**: Deploy to production with monitoring and MLOps
5. **Phase 5**: Polish frontend and user experience

**Advanced Features (Phases 6-7):** 
- Implement after core functionality is stable and deployed
- **Phase 6**: Add DPO for preference learning (advanced technique)
- **Phase 7**: Add comprehensive evaluation framework (research validation)

> **Recommendation**: Focus on Phases 1-5 first to get a working, deployed system. Phases 6-7 can be added incrementally based on time and research needs.

---

## Key Areas Requiring Additional Attention

### 1. **Data Privacy & Security**
- Implement secure file handling for uploaded research papers
- Add API rate limiting and authentication
- Ensure GDPR/privacy compliance for user data

### 2. **Model Licensing & Compliance**
- Verify licensing for fine-tuned model redistribution
- Implement proper attribution for training datasets
- Consider commercial vs. research use implications

### 3. **Scalability Considerations**
- Database optimization for large document collections
- Caching strategies for frequently accessed papers
- Load balancing for multiple model inference

### 4. **Quality Assurance**
- Comprehensive testing suite (unit, integration, end-to-end)
- Model performance regression testing
- Citation accuracy validation pipeline

### 5. **Documentation & Reproducibility**
- Detailed API documentation with OpenAPI/Swagger
- Model training and evaluation reproducibility
- Deployment guides and architecture diagrams