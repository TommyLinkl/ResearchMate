#!/usr/bin/env python3
"""
LoRA Fine-tuning script for ResearchMate-Mistral-7B
Supports multi-GPU training with DeepSpeed and gradient checkpointing
"""
import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from datasets import Dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import (
    load_model_and_tokenizer,
    create_lora_config, 
    prepare_model_for_training,
    create_training_arguments,
    save_model_and_adapters,
    get_model_size_info
)
from utils.metrics import ModelEvaluator, calculate_training_metrics
from utils.preprocessing import load_config, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchMateTrainer:
    """Custom trainer for ResearchMate fine-tuning"""
    
    def __init__(
        self, 
        model_config_path: str,
        training_config_path: str,
        data_config_path: str,
        output_dir: str
    ):
        self.model_config = load_config(model_config_path)
        self.training_config = load_config(training_config_path)
        self.data_config = load_config(data_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.evaluator = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        
        logger.info("Setting up model and tokenizer...")
        
        base_model_name = self.model_config['model']['base_model']
        use_4bit = self.model_config['quantization']['use_4bit']
        
        # Load base model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            base_model_name, 
            use_4bit=use_4bit
        )
        
        # Create LoRA configuration
        lora_config = create_lora_config("training/configs/lora_config.yaml")
        
        # Prepare model for training
        self.model = prepare_model_for_training(
            self.model, 
            lora_config,
            use_gradient_checkpointing=self.training_config['training']['gradient_checkpointing']
        )
        
        # Get model info
        model_info = get_model_size_info(self.model)
        logger.info(f"Model info: {model_info}")
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(base_model_name)
        
    def load_and_prepare_datasets(self) -> tuple:
        """Load and prepare training datasets"""
        
        logger.info("Loading datasets...")
        
        data_dir = Path(self.data_config['paths']['processed_data'])
        
        # Load datasets
        train_data = load_dataset(data_dir / "train_dataset.json")
        val_data = load_dataset(data_dir / "val_dataset.json")
        
        logger.info(f"Loaded {len(train_data)} training samples")
        logger.info(f"Loaded {len(val_data)} validation samples")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.data_config['data']['processing']['max_length']
            )
        
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["instruction", "input", "output"]
        )
        
        val_dataset = val_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["instruction", "input", "output"]
        )
        
        return train_dataset, val_dataset
    
    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset):
        """Setup the trainer with all configurations"""
        
        logger.info("Setting up trainer...")
        
        # Create training arguments
        training_args = create_training_arguments(
            "training/configs/training_params.yaml",
            str(self.output_dir)
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.training_config['training']['early_stopping_patience'],
            early_stopping_threshold=self.training_config['training']['early_stopping_threshold']
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )
        
    def train(self):
        """Execute the training process"""
        
        logger.info("Starting training...")
        
        # Check for existing checkpoints
        checkpoint_dir = None
        if (self.output_dir / "checkpoint-*").exists():
            checkpoints = list(self.output_dir.glob("checkpoint-*"))
            if checkpoints:
                checkpoint_dir = str(max(checkpoints, key=os.path.getctime))
                logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
        
        # Train the model
        training_result = self.trainer.train(resume_from_checkpoint=checkpoint_dir)
        
        # Save training metrics
        training_metrics = calculate_training_metrics(self.trainer.state)
        training_metrics.update({
            "training_runtime": training_result.metrics.get("train_runtime", 0),
            "training_samples_per_second": training_result.metrics.get("train_samples_per_second", 0),
            "training_steps_per_second": training_result.metrics.get("train_steps_per_second", 0),
            "total_flos": training_result.metrics.get("total_flos", 0),
            "train_loss": training_result.metrics.get("train_loss", 0)
        })
        
        logger.info("Training completed!")
        return training_metrics
    
    def evaluate_model(self, test_data_path: str = None) -> Dict[str, Any]:
        """Evaluate the trained model"""
        
        logger.info("Evaluating model...")
        
        # Load test data
        if test_data_path:
            test_data = load_dataset(test_data_path)
        else:
            # Use validation data for evaluation
            data_dir = Path(self.data_config['paths']['processed_data'])
            test_data = load_dataset(data_dir / "val_dataset.json")
        
        # Evaluate physics Q&A performance
        results = self.evaluator.evaluate_physics_qa(
            self.model, 
            self.tokenizer, 
            test_data
        )
        
        # Calculate perplexity
        try:
            perplexity = self.evaluator.compute_perplexity(self.model, test_data)
            results["perplexity"] = perplexity
        except Exception as e:
            logger.warning(f"Could not calculate perplexity: {e}")
            results["perplexity"] = None
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_final_model(self, training_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        """Save the final model and all artifacts"""
        
        logger.info("Saving final model...")
        
        # Combine all metrics
        final_metrics = {
            "training_metrics": training_metrics,
            "evaluation_results": eval_results,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "timestamp": datetime.now().isoformat(),
            "model_info": get_model_size_info(self.model)
        }
        
        # Save to fine_tuned_models directory
        final_model_dir = Path("fine_tuned_models/ResearchMate-Mistral-7B")
        
        save_model_and_adapters(
            self.model,
            self.tokenizer,
            str(final_model_dir),
            final_metrics
        )
        
        logger.info(f"Final model saved to: {final_model_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResearchMate-Mistral-7B with LoRA")
    parser.add_argument("--model-config", default="training/configs/lora_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--training-config", default="training/configs/training_params.yaml", 
                       help="Path to training configuration")
    parser.add_argument("--data-config", default="training/configs/data_config.yaml",
                       help="Path to data configuration")
    parser.add_argument("--output-dir", default="training/models",
                       help="Output directory for training artifacts")
    parser.add_argument("--test-data", default=None,
                       help="Path to test dataset for evaluation")
    parser.add_argument("--resume-from-checkpoint", default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup distributed training if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])
    
    # Initialize trainer
    trainer = ResearchMateTrainer(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        data_config_path=args.data_config,
        output_dir=args.output_dir
    )
    
    try:
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        # Load and prepare datasets
        train_dataset, val_dataset = trainer.load_and_prepare_datasets()
        
        # Setup trainer
        trainer.setup_trainer(train_dataset, val_dataset)
        
        # Train the model
        training_metrics = trainer.train()
        
        # Evaluate the model
        eval_results = trainer.evaluate_model(args.test_data)
        
        # Save final model
        trainer.save_final_model(training_metrics, eval_results)
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()