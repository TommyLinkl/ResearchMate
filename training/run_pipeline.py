#!/usr/bin/env python3
"""
Complete pipeline runner for ResearchMate Phase 3 fine-tuning
Orchestrates data preparation, training, and evaluation
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: str, description: str):
    """Run shell command with logging"""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {cmd}")
    
    start_time = datetime.now()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = datetime.now()
    duration = end_time - start_time
    
    if result.returncode == 0:
        logger.info(f"✅ Completed: {description} (Duration: {duration})")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
    else:
        logger.error(f"❌ Failed: {description}")
        logger.error(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)

def main():
    parser = argparse.ArgumentParser(description="Run ResearchMate fine-tuning pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip final evaluation")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting ResearchMate Phase 3 Fine-tuning Pipeline")
    
    # Change to training directory
    os.chdir("training")
    
    try:
        # Step 1: Data Preparation
        if not args.skip_data:
            run_command(
                "python scripts/data_preparation.py",
                "Data preparation and processing"
            )
        
        # Step 2: Baseline Evaluation  
        if not args.skip_baseline:
            run_command(
                "python scripts/baseline_eval.py",
                "Baseline model evaluation"
            )
        
        # Step 3: Fine-tuning
        if not args.skip_training:
            if args.gpus and args.gpus > 1:
                cmd = f"torchrun --nproc_per_node={args.gpus} scripts/train_lora.py"
            else:
                cmd = "python scripts/train_lora.py"
            
            run_command(cmd, "Model fine-tuning with LoRA")
        
        # Step 4: Model Evaluation
        if not args.skip_evaluation:
            # Evaluate fine-tuned model
            run_command(
                "python scripts/evaluate_model.py --mode evaluate",
                "Fine-tuned model evaluation"
            )
            
            # Compare models
            run_command(
                "python scripts/evaluate_model.py --mode compare", 
                "Model performance comparison"
            )
            
            # Generate report
            run_command(
                "python scripts/evaluate_model.py --mode report",
                "Evaluation report generation"
            )
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Check training/data/evaluation/ for results")
        logger.info("🤖 Fine-tuned model saved in fine_tuned_models/ResearchMate-Mistral-7B/")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"💥 Pipeline failed at: {e.cmd}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("⏸️  Pipeline interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()