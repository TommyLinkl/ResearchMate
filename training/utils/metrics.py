"""
Evaluation metrics utilities for ResearchMate fine-tuning
"""
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities"""
    
    def __init__(self, tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_perplexity(self, model, dataset, max_length: int = 2048) -> float:
        """Calculate perplexity on evaluation dataset"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for item in dataset:
                text = item['text']
                inputs = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Count tokens for proper averaging
                mask = inputs['attention_mask']
                num_tokens = mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_physics_qa(self, model, tokenizer, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate model on physics Q&A tasks"""
        
        results = {
            "accuracy": 0.0,
            "avg_response_length": 0.0,
            "physics_keywords_coverage": 0.0
        }
        
        physics_keywords = [
            "quantum", "energy", "momentum", "force", "field", "particle",
            "wave", "electron", "photon", "atom", "molecule", "temperature",
            "pressure", "entropy", "entanglement", "superposition"
        ]
        
        correct_responses = 0
        total_response_length = 0
        keyword_coverage_scores = []
        
        model.eval()
        
        with torch.no_grad():
            for item in test_data:
                instruction = item['instruction']
                expected_output = item['output']
                
                # Generate response
                prompt = f"<s>[INST] {instruction} [/INST]"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                generated = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                response = response.replace(prompt.replace("<s>", "").replace("</s>", ""), "").strip()
                
                # Simple correctness check (contains key physics terms)
                response_lower = response.lower()
                expected_lower = expected_output.lower()
                
                # Check if response contains relevant physics keywords
                keywords_in_response = sum(1 for kw in physics_keywords if kw in response_lower)
                keywords_in_expected = sum(1 for kw in physics_keywords if kw in expected_lower)
                
                if keywords_in_expected > 0:
                    keyword_coverage = keywords_in_response / keywords_in_expected
                    keyword_coverage_scores.append(min(keyword_coverage, 1.0))
                
                # Length tracking
                total_response_length += len(response)
                
                # Simple correctness heuristic
                if len(response) > 20 and keywords_in_response > 0:
                    correct_responses += 1
        
        results["accuracy"] = correct_responses / len(test_data) if test_data else 0.0
        results["avg_response_length"] = total_response_length / len(test_data) if test_data else 0.0
        results["physics_keywords_coverage"] = np.mean(keyword_coverage_scores) if keyword_coverage_scores else 0.0
        
        return results
    
    def compare_models(self, base_results: Dict[str, float], finetuned_results: Dict[str, float]) -> Dict[str, float]:
        """Compare base model vs fine-tuned model performance"""
        
        comparison = {}
        
        for metric in base_results:
            if metric in finetuned_results:
                base_val = base_results[metric]
                ft_val = finetuned_results[metric]
                
                if base_val != 0:
                    improvement = ((ft_val - base_val) / base_val) * 100
                else:
                    improvement = 100.0 if ft_val > 0 else 0.0
                
                comparison[f"{metric}_improvement_pct"] = improvement
                comparison[f"base_{metric}"] = base_val
                comparison[f"finetuned_{metric}"] = ft_val
        
        return comparison

def save_evaluation_results(results: Dict[str, Any], filepath: str):
    """Save evaluation results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_training_metrics(trainer_state) -> Dict[str, float]:
    """Extract training metrics from trainer state"""
    
    if not trainer_state.log_history:
        return {}
    
    metrics = {
        "final_train_loss": None,
        "final_eval_loss": None,
        "best_eval_loss": float('inf'),
        "total_training_steps": trainer_state.global_step,
        "training_epochs": trainer_state.epoch
    }
    
    # Extract metrics from log history
    for log_entry in trainer_state.log_history:
        if 'train_loss' in log_entry:
            metrics["final_train_loss"] = log_entry['train_loss']
        if 'eval_loss' in log_entry:
            metrics["final_eval_loss"] = log_entry['eval_loss']
            if log_entry['eval_loss'] < metrics["best_eval_loss"]:
                metrics["best_eval_loss"] = log_entry['eval_loss']
    
    return metrics