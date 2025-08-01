import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM instances - focused on 3 core models for research use"""
    
    def __init__(self):
        self.llms: Dict[str, any] = {}
        self.active_model = "Llama-3.1-8B"  # Default to open-source model
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the 3 core LLM models"""
        
        # Model 1: Llama 3.1 8B (Primary open-source model)
        try:
            if os.getenv('GROQ_API_KEY'):
                from langchain_groq import ChatGroq
                self.llms["Llama-3.1-8B"] = ChatGroq(
                    temperature=0.6, 
                    model_name='llama-3.1-8b-instant', 
                    groq_api_key=os.getenv('GROQ_API_KEY')
                )
                logger.info("Llama-3.1-8B initialized successfully")
            else:
                logger.warning("Groq API key not found, Llama-3.1-8B unavailable")
        except Exception as e:
            logger.error(f"Error initializing Llama-3.1-8B: {e}")
        
        # Model 2: Mistral-7B (Secondary open-source model)
        try:
            if os.getenv('GROQ_API_KEY'):
                from langchain_groq import ChatGroq
                self.llms["Mistral-7B"] = ChatGroq(
                    temperature=0.6,
                    model_name='mixtral-8x7b-32768',  # Using Mixtral as Mistral-7B alternative
                    groq_api_key=os.getenv('GROQ_API_KEY')
                )
                logger.info("Mistral-7B initialized successfully")
            else:
                logger.warning("Groq API key not found, Mistral-7B unavailable")
        except Exception as e:
            logger.error(f"Error initializing Mistral-7B: {e}")
        
        # Model 3: Fine-tuned Mistral-7B (To be implemented in Phase 2/3)
        self.llms["ResearchMate-Mistral-7B"] = None  # Placeholder for fine-tuned model
        logger.info("ResearchMate-Mistral-7B: Placeholder for fine-tuned model (Phase 2/3)")
        
        # Set default active model to first available
        available_models = [name for name, model in self.llms.items() if model is not None]
        if available_models:
            self.active_model = available_models[0]
            logger.info(f"Active model set to: {self.active_model}")
        else:
            logger.error("No LLM models available!")
    
    def get_llm(self, model_name: Optional[str] = None) -> any:
        """Get LLM instance by name, or return active model"""
        if model_name is None:
            model_name = self.active_model
        
        if model_name not in self.llms:
            available = [name for name, model in self.llms.items() if model is not None]
            raise ValueError(f"Model '{model_name}' not available. Available models: {available}")
        
        model = self.llms[model_name]
        if model is None:
            raise ValueError(f"Model '{model_name}' is not yet implemented (coming in future phases)")
        
        return model
    
    def set_active_model(self, model_name: str):
        """Set the active model"""
        if model_name not in self.llms:
            available = [name for name, model in self.llms.items() if model is not None]
            raise ValueError(f"Model '{model_name}' not available. Available models: {available}")
        
        if self.llms[model_name] is None:
            raise ValueError(f"Model '{model_name}' is not yet implemented (coming in future phases)")
        
        self.active_model = model_name
        logger.info(f"Active model changed to: {model_name}")
    
    def get_active_model(self) -> str:
        """Get the name of the active model"""
        return self.active_model
    
    def get_available_models(self) -> List[Dict[str, any]]:
        """Get list of available models with their info"""
        models = []
        for name, llm in self.llms.items():
            if name == "ResearchMate-Mistral-7B":
                models.append({
                    "name": name,
                    "type": "Fine-tuned Physics Model",
                    "available": False,
                    "description": "Coming in Phase 2/3 - Fine-tuned Mistral-7B for physics research"
                })
            elif llm is not None:
                model_type = "Foundation Model"
                description = "Open-source foundation model"
                if "Llama" in name:
                    description = "Meta's Llama model - excellent for general reasoning"
                elif "Mistral" in name:
                    description = "Mistral AI model - optimized for efficiency and performance"
                
                models.append({
                    "name": name,
                    "type": model_type,
                    "available": True,
                    "description": description
                })
        return models
    
    def run_llm(self, query: str, llm: any = None) -> str:
        """Run a query through the specified LLM or active model"""
        if llm is None:
            llm = self.get_llm()
        
        try:
            messages = [{"role": "user", "content": query}]
            response = llm.invoke(input=messages)
            return response.content
        except Exception as e:
            logger.error(f"Error running LLM query: {e}")
            raise
    
    def add_model(self, name: str, model_instance: any):
        """Add a new model instance (useful for fine-tuned models later)"""
        self.llms[name] = model_instance
        logger.info(f"Added new model: {name}")
    
    def remove_model(self, name: str):
        """Remove a model instance"""
        if name in self.llms:
            del self.llms[name]
            if self.active_model == name and self.llms:
                self.active_model = list(self.llms.keys())[0]
            logger.info(f"Removed model: {name}")
        else:
            logger.warning(f"Model '{name}' not found for removal")