"""
Model utilities for loading, saving, and managing models
"""
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
import yaml
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    trust_remote_code: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer with optional quantization"""
    
    logger.info(f"Loading model: {model_name}")
    
    # Configure quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_4bit else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

def create_lora_config(config_path: str) -> LoraConfig:
    """Create LoRA configuration from YAML file"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lora_params = config['lora']
    
    return LoraConfig(
        r=lora_params['r'],
        lora_alpha=lora_params['lora_alpha'],
        lora_dropout=lora_params['lora_dropout'],
        target_modules=lora_params['target_modules'],
        bias=lora_params['bias'],
        task_type=lora_params['task_type']
    )

def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig,
    use_gradient_checkpointing: bool = True
) -> AutoModelForCausalLM:
    """Prepare model for LoRA fine-tuning"""
    
    logger.info("Preparing model for training...")
    
    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare for k-bit training if using quantization
    if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def create_training_arguments(config_path: str, output_dir: str) -> TrainingArguments:
    """Create training arguments from configuration"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_params = config['training']
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_params['num_train_epochs'],
        per_device_train_batch_size=training_params['per_device_train_batch_size'],
        per_device_eval_batch_size=training_params['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_params['gradient_accumulation_steps'],
        gradient_checkpointing=training_params['gradient_checkpointing'],
        learning_rate=training_params['learning_rate'],
        lr_scheduler_type=training_params['lr_scheduler_type'],
        warmup_ratio=training_params['warmup_ratio'],
        weight_decay=training_params['weight_decay'],
        optim=training_params['optim'],
        fp16=training_params['fp16'],
        bf16=training_params['bf16'],
        logging_steps=training_params['logging_steps'],
        save_steps=training_params['save_steps'],
        eval_steps=training_params['eval_steps'],
        save_total_limit=training_params['save_total_limit'],
        load_best_model_at_end=training_params['load_best_model_at_end'],
        metric_for_best_model=training_params['metric_for_best_model'],
        greater_is_better=training_params['greater_is_better'],
        dataloader_num_workers=training_params['dataloader_num_workers'],
        ddp_find_unused_parameters=training_params['ddp_find_unused_parameters'],
        evaluation_strategy=config['evaluation']['evaluation_strategy'],
        do_eval=config['evaluation']['do_eval'],
        eval_accumulation_steps=config['evaluation']['eval_accumulation_steps'],
        report_to=None,  # Disable wandb/tensorboard for now
        run_name="researchmate-mistral-7b"
    )

def save_model_and_adapters(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    training_metrics: Dict[str, Any] = None
):
    """Save the fine-tuned model adapters and tokenizer"""
    
    logger.info(f"Saving model to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics if provided
    if training_metrics:
        import json
        with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
            json.dump(training_metrics, f, indent=2)
    
    logger.info("Model saved successfully!")

def load_fine_tuned_model(
    base_model_name: str,
    adapter_path: str,
    use_4bit: bool = True
) -> Tuple[PeftModel, AutoTokenizer]:
    """Load fine-tuned model with adapters for inference"""
    
    logger.info(f"Loading fine-tuned model from {adapter_path}")
    
    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_name, 
        use_4bit=use_4bit
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def get_model_size_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """Get model size and parameter information"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100,
        "model_size_mb": (total_params * 4) / (1024 * 1024),  # Assuming float32
        "memory_footprint_gb": None
    }
    
    # Try to get actual memory usage
    if torch.cuda.is_available():
        try:
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            info["memory_footprint_gb"] = memory_mb / 1024
        except:
            pass
    
    return info