"""
Model comparison utilities for ResearchMate
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_comparison_chart(
    baseline_results: Dict[str, Any],
    finetuned_results: Dict[str, Any],
    output_path: str
):
    """Create comparison chart between models"""
    
    metrics = ['accuracy', 'physics_keywords_coverage', 'avg_response_length']
    baseline_values = [baseline_results.get(m, 0) for m in metrics]
    finetuned_values = [finetuned_results.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison chart saved to: {output_path}")

def generate_improvement_summary(
    baseline_results: Dict[str, Any],
    finetuned_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate improvement summary between models"""
    
    summary = {
        "improvements": {},
        "recommendations": []
    }
    
    key_metrics = ['accuracy', 'physics_keywords_coverage']
    
    for metric in key_metrics:
        baseline_val = baseline_results.get(metric, 0)
        finetuned_val = finetuned_results.get(metric, 0)
        
        if baseline_val > 0:
            improvement_pct = ((finetuned_val - baseline_val) / baseline_val) * 100
        else:
            improvement_pct = 100 if finetuned_val > 0 else 0
        
        summary["improvements"][metric] = {
            "baseline": baseline_val,
            "finetuned": finetuned_val,
            "improvement_pct": improvement_pct,
            "improved": improvement_pct > 0
        }
    
    # Generate recommendations
    avg_improvement = np.mean([v["improvement_pct"] for v in summary["improvements"].values()])
    
    if avg_improvement >= 15:
        summary["recommendations"].append("Excellent improvement! Model is ready for production deployment.")
    elif avg_improvement >= 5:
        summary["recommendations"].append("Good improvement. Consider additional training or data augmentation.")
    else:
        summary["recommendations"].append("Limited improvement. Review training data quality and hyperparameters.")
    
    # Specific metric recommendations
    accuracy_improvement = summary["improvements"]["accuracy"]["improvement_pct"]
    if accuracy_improvement < 5:
        summary["recommendations"].append("Low accuracy improvement suggests need for more diverse training data.")
    
    physics_improvement = summary["improvements"]["physics_keywords_coverage"]["improvement_pct"]
    if physics_improvement < 10:
        summary["recommendations"].append("Physics knowledge improvement is limited. Consider domain-specific data augmentation.")
    
    return summary

def compare_models_comprehensive(
    baseline_path: str,
    finetuned_path: str,
    output_dir: str
):
    """Comprehensive model comparison with visualizations and reports"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    baseline_results = load_results(baseline_path)
    finetuned_results = load_results(finetuned_path)
    
    # Create comparison chart
    chart_path = output_dir / "model_comparison_chart.png"
    create_comparison_chart(baseline_results, finetuned_results, str(chart_path))
    
    # Generate improvement summary
    summary = generate_improvement_summary(baseline_results, finetuned_results)
    
    # Save comprehensive comparison
    comparison_results = {
        "baseline_results": baseline_results,
        "finetuned_results": finetuned_results,
        "improvement_summary": summary,
        "charts": {
            "comparison_chart": str(chart_path)
        }
    }
    
    comparison_path = output_dir / "comprehensive_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Generate markdown report
    report_lines = [
        "# ResearchMate Model Comparison Report",
        "",
        "## Key Improvements",
        ""
    ]
    
    for metric, data in summary["improvements"].items():
        status = "✅" if data["improved"] else "❌"
        report_lines.extend([
            f"**{metric.replace('_', ' ').title()}:** {status}",
            f"- Baseline: {data['baseline']:.3f}",
            f"- Fine-tuned: {data['finetuned']:.3f}",
            f"- Improvement: {data['improvement_pct']:.1f}%",
            ""
        ])
    
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    for rec in summary["recommendations"]:
        report_lines.append(f"- {rec}")
    
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Comprehensive comparison completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Average improvement: {np.mean([v['improvement_pct'] for v in summary['improvements'].values()]):.1f}%")

if __name__ == "__main__":
    # Example usage
    compare_models_comprehensive(
        "training/data/evaluation/baseline_results.json",
        "training/data/evaluation/finetuned_results.json", 
        "training/data/evaluation/comparison"
    )