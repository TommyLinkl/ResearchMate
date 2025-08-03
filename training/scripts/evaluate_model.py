#!/usr/bin/env python3
"""
Model evaluation and comparison script for ResearchMate
Compares base model vs fine-tuned model performance
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

from utils.model_utils import load_model_and_tokenizer, load_fine_tuned_model
from utils.metrics import ModelEvaluator, save_evaluation_results, load_evaluation_results
from utils.preprocessing import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_fine_tuned_model(
    base_model_name: str,
    adapter_path: str,
    test_data_path: str,
    output_path: str,
    use_4bit: bool = True
):
    """Evaluate fine-tuned model performance"""
    
    logger.info(f"Evaluating fine-tuned model: {adapter_path}")
    
    # Load fine-tuned model
    model, tokenizer = load_fine_tuned_model(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        use_4bit=use_4bit
    )
    
    # Load test data
    test_data = load_dataset(test_data_path)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(base_model_name)
    
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
    results["base_model_name"] = base_model_name
    results["adapter_path"] = adapter_path
    results["test_data_path"] = test_data_path
    results["test_samples"] = len(test_data)
    results["timestamp"] = datetime.now().isoformat()
    results["evaluation_type"] = "fine_tuned"
    
    # Save results
    save_evaluation_results(results, output_path)
    
    logger.info(f"Fine-tuned model evaluation completed!")
    logger.info(f"Results saved to: {output_path}")
    
    return results

def compare_models(
    baseline_results_path: str,
    finetuned_results_path: str,
    output_path: str
):
    """Compare baseline vs fine-tuned model performance"""
    
    logger.info("Comparing model performances...")
    
    # Load results
    baseline_results = load_evaluation_results(baseline_results_path)
    finetuned_results = load_evaluation_results(finetuned_results_path)
    
    # Initialize evaluator for comparison
    evaluator = ModelEvaluator()
    
    # Compare models
    comparison = evaluator.compare_models(baseline_results, finetuned_results)
    
    # Add metadata
    comparison_results = {
        "baseline_results": baseline_results,
        "finetuned_results": finetuned_results,
        "improvements": comparison,
        "timestamp": datetime.now().isoformat(),
        "comparison_summary": {
            "accuracy_improved": comparison.get("accuracy_improvement_pct", 0) > 0,
            "physics_coverage_improved": comparison.get("physics_keywords_coverage_improvement_pct", 0) > 0,
            "overall_improvement": comparison.get("accuracy_improvement_pct", 0) > 10  # 10% threshold
        }
    }
    
    # Save comparison results
    save_evaluation_results(comparison_results, output_path)
    
    logger.info("Model comparison completed!")
    logger.info(f"Comparison results saved to: {output_path}")
    
    return comparison_results

def generate_evaluation_report(comparison_results: dict, output_path: str):
    """Generate a human-readable evaluation report"""
    
    report_lines = [
        "# ResearchMate Model Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Performance Comparison",
        ""
    ]
    
    # Extract key metrics
    baseline = comparison_results["baseline_results"]
    finetuned = comparison_results["finetuned_results"]
    improvements = comparison_results["improvements"]
    
    # Accuracy comparison
    report_lines.extend([
        f"**Accuracy:**",
        f"- Baseline: {baseline['accuracy']:.3f}",
        f"- Fine-tuned: {finetuned['accuracy']:.3f}",
        f"- Improvement: {improvements.get('accuracy_improvement_pct', 0):.1f}%",
        ""
    ])
    
    # Physics coverage comparison
    report_lines.extend([
        f"**Physics Keywords Coverage:**",
        f"- Baseline: {baseline['physics_keywords_coverage']:.3f}",
        f"- Fine-tuned: {finetuned['physics_keywords_coverage']:.3f}",
        f"- Improvement: {improvements.get('physics_keywords_coverage_improvement_pct', 0):.1f}%",
        ""
    ])
    
    # Perplexity comparison (if available)
    if baseline.get('perplexity') and finetuned.get('perplexity'):
        report_lines.extend([
            f"**Perplexity:**",
            f"- Baseline: {baseline['perplexity']:.2f}",
            f"- Fine-tuned: {finetuned['perplexity']:.2f}",
            f"- Improvement: {improvements.get('perplexity_improvement_pct', 0):.1f}%",
            ""
        ])
    
    # Summary
    summary = comparison_results["comparison_summary"]
    report_lines.extend([
        "## Summary",
        "",
        f"- Accuracy improved: {'✅' if summary['accuracy_improved'] else '❌'}",
        f"- Physics coverage improved: {'✅' if summary['physics_coverage_improved'] else '❌'}",
        f"- Overall improvement (>10%): {'✅' if summary['overall_improvement'] else '❌'}",
        ""
    ])
    
    # Recommendations
    if summary['overall_improvement']:
        report_lines.append("**Recommendation:** Fine-tuned model shows significant improvement and is ready for deployment.")
    else:
        report_lines.append("**Recommendation:** Fine-tuned model shows limited improvement. Consider adjusting training parameters or data quality.")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Evaluation report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--mode", choices=["evaluate", "compare", "report"], required=True,
                       help="Evaluation mode")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="Base model name")
    parser.add_argument("--adapter-path", default="fine_tuned_models/ResearchMate-Mistral-7B",
                       help="Path to fine-tuned model adapters")
    parser.add_argument("--test-data", default="training/data/processed/val_dataset.json",
                       help="Path to test dataset")
    parser.add_argument("--baseline-results", default="training/data/evaluation/baseline_results.json",
                       help="Path to baseline results")
    parser.add_argument("--finetuned-results", default="training/data/evaluation/finetuned_results.json",
                       help="Path to fine-tuned results")
    parser.add_argument("--comparison-output", default="training/data/evaluation/model_comparison.json",
                       help="Output path for comparison results")
    parser.add_argument("--report-output", default="training/data/evaluation/evaluation_report.md",
                       help="Output path for evaluation report")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Create output directories
    for path in [args.finetuned_results, args.comparison_output, args.report_output]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "evaluate":
        # Evaluate fine-tuned model
        results = evaluate_fine_tuned_model(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            test_data_path=args.test_data,
            output_path=args.finetuned_results,
            use_4bit=not args.no_4bit
        )
        
        print(f"\n=== Fine-tuned Model Results ===")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Physics Keywords Coverage: {results['physics_keywords_coverage']:.3f}")
        print(f"Average Response Length: {results['avg_response_length']:.1f}")
        if results.get('perplexity'):
            print(f"Perplexity: {results['perplexity']:.2f}")
    
    elif args.mode == "compare":
        # Compare models
        comparison_results = compare_models(
            baseline_results_path=args.baseline_results,
            finetuned_results_path=args.finetuned_results,
            output_path=args.comparison_output
        )
        
        improvements = comparison_results["improvements"]
        print(f"\n=== Model Comparison ===")
        print(f"Accuracy improvement: {improvements.get('accuracy_improvement_pct', 0):.1f}%")
        print(f"Physics coverage improvement: {improvements.get('physics_keywords_coverage_improvement_pct', 0):.1f}%")
        
    elif args.mode == "report":
        # Generate report
        comparison_results = load_evaluation_results(args.comparison_output)
        generate_evaluation_report(comparison_results, args.report_output)
        print(f"Evaluation report generated: {args.report_output}")

if __name__ == "__main__":
    main()