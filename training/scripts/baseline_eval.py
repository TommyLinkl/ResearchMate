#!/usr/bin/env python3
"""
Baseline evaluation script for ResearchMate
Establishes baseline performance metrics before fine-tuning
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import load_model_and_tokenizer
from utils.metrics import ModelEvaluator, save_evaluation_results
from utils.preprocessing import load_config, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_baseline_evaluation(
    model_name: str,
    test_data_path: str,
    output_path: str,
    use_4bit: bool = True
):
    """Run baseline evaluation on the base model"""
    
    logger.info(f"Running baseline evaluation for {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit=use_4bit)
    
    # Load test data
    test_data = load_dataset(test_data_path)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_name)
    
    # Run physics Q&A evaluation
    results = evaluator.evaluate_physics_qa(model, tokenizer, test_data)
    
    # Calculate perplexity
    try:
        perplexity = evaluator.compute_perplexity(model, test_data)
        results["perplexity"] = perplexity
    except Exception as e:
        logger.warning(f"Could not calculate perplexity: {e}")
        results["perplexity"] = None
    
    # Add metadata
    results["model_name"] = model_name
    results["test_data_path"] = test_data_path
    results["test_samples"] = len(test_data)
    results["timestamp"] = datetime.now().isoformat()
    results["evaluation_type"] = "baseline"
    
    # Save results
    save_evaluation_results(results, output_path)
    
    logger.info(f"Baseline evaluation completed!")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Baseline accuracy: {results['accuracy']:.3f}")
    logger.info(f"Baseline perplexity: {results.get('perplexity', 'N/A')}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="Base model name")
    parser.add_argument("--test-data", default="training/data/processed/val_dataset.json",
                       help="Path to test dataset")
    parser.add_argument("--output", default="training/data/evaluation/baseline_results.json",
                       help="Output path for results")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results = run_baseline_evaluation(
        model_name=args.model,
        test_data_path=args.test_data,
        output_path=str(output_path),
        use_4bit=not args.no_4bit
    )
    
    print(f"\n=== Baseline Results ===")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Physics Keywords Coverage: {results['physics_keywords_coverage']:.3f}")
    print(f"Average Response Length: {results['avg_response_length']:.1f}")
    if results.get('perplexity'):
        print(f"Perplexity: {results['perplexity']:.2f}")

if __name__ == "__main__":
    main()