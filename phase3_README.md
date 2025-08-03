# Phase 3: Domain-Specific Fine-Tuning Pipeline

Complete implementation of ResearchMate-Mistral-7B fine-tuning with LoRA adapters.

## Overview

This phase implements a production-ready fine-tuning pipeline that creates ResearchMate-Mistral-7B, a physics-specialized model optimized for scientific question-answering.

**Key Features:**
- LoRA/QLoRA fine-tuning with 4-bit quantization
- Multi-GPU training support (up to 8x NVIDIA A100)
- Automated data preparation from Physics Stack Exchange & ArXiv
- Comprehensive evaluation and model comparison framework
- Risk mitigation with checkpointing and fallback strategies

## Quick Start

### 1. Setup Environment
```bash
cd training/
pip install -r requirements.txt
```

### 2. Prepare Data (1-2 days automated)
```bash
python scripts/data_preparation.py
# Creates ~8K physics Q&A examples from Stack Exchange + ArXiv
```

### 3. Baseline Evaluation 
```bash
python scripts/baseline_eval.py --model mistralai/Mistral-7B-Instruct-v0.1
# Establishes performance metrics before fine-tuning
```

### 4. Fine-tune Model (2-3 days on 8x A100)
```bash
# Single GPU
python scripts/train_lora.py

# Multi-GPU (automatic detection)
torchrun --nproc_per_node=8 scripts/train_lora.py
```

### 5. Evaluate & Compare
```bash
# Evaluate fine-tuned model
python scripts/evaluate_model.py --mode evaluate

# Compare with baseline
python scripts/evaluate_model.py --mode compare

# Generate report
python scripts/evaluate_model.py --mode report
```

## Directory Structure

```
training/
├── configs/
│   ├── data_config.yaml      # Dataset processing settings
│   ├── lora_config.yaml      # LoRA hyperparameters  
│   └── training_params.yaml  # Training configuration
├── data/
│   ├── raw/                  # Downloaded datasets (git ignored)
│   ├── processed/            # Training-ready data (git ignored)
│   └── evaluation/           # Test sets & results (git ignored)
├── scripts/
│   ├── data_preparation.py   # Automated data processing
│   ├── train_lora.py        # Main training script
│   ├── baseline_eval.py     # Pre-training evaluation
│   └── evaluate_model.py    # Model comparison
├── utils/
│   ├── preprocessing.py     # Data processing utilities
│   ├── metrics.py          # Evaluation metrics
│   └── model_utils.py      # Model loading/saving
├── evaluation/
│   ├── benchmarks.py       # Physics Q&A benchmarks
│   └── comparison.py       # Model comparison tools
└── models/                 # Training checkpoints (git ignored)

fine_tuned_models/          # Only deliverable tracked in git
└── ResearchMate-Mistral-7B/
    ├── adapter_config.json
    ├── adapter_model.safetensors  # Final LoRA adapters (~20MB)
    └── training_metrics.json
```

## Configuration

### LoRA Parameters (Optimized for A100)
- **Rank (r)**: 16 (balance between performance & efficiency)
- **Alpha**: 32 (2x rank for stability)
- **Target Modules**: All attention & MLP layers
- **Dropout**: 0.1 for regularization

### Training Settings (8x A100 Setup)
- **Batch Size**: 4 per device (32 total with 8 GPUs)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4 with cosine scheduling
- **Epochs**: 3 (with early stopping)
- **Quantization**: 4-bit QLoRA for memory efficiency

### Dataset Strategy
- **Primary**: Physics Stack Exchange Q&A (5K examples)
- **Secondary**: ArXiv abstracts → Q&A (3K examples)
- **Total**: 8K examples (sufficient for LoRA)
- **Split**: 90% train, 10% validation

## Key Scripts

### data_preparation.py
Automated data pipeline:
- Downloads Physics Stack Exchange dump
- Extracts ArXiv physics papers
- Applies quality filters
- Formats for instruction tuning
- Generates training/validation splits

### train_lora.py
Main training script with:
- Multi-GPU support via PyTorch DDP
- 4-bit quantization for memory efficiency
- Gradient checkpointing
- Early stopping
- Automatic checkpoint resumption

### evaluate_model.py
Comprehensive evaluation:
- Physics-specific metrics (keyword coverage, accuracy)
- Perplexity calculation
- Model comparison with statistical significance
- Automated report generation

## Multi-GPU Training

**Automatic GPU Detection:**
```bash
# Detects available GPUs automatically
python scripts/train_lora.py
```

**Manual Multi-GPU Setup:**
```bash
# 8 GPU training
torchrun --nproc_per_node=8 scripts/train_lora.py

# 4 GPU training  
torchrun --nproc_per_node=4 scripts/train_lora.py
```

**Memory Requirements:**
- Single A100 (80GB): Full model + LoRA adapters
- 4x A100: Distributed training with gradient accumulation
- 8x A100: Optimal performance with 4 batch size per GPU

## Success Criteria

**Minimum Requirements:**
- **Accuracy improvement**: >10% over baseline
- **Physics coverage**: >15% improvement in domain keywords
- **Training stability**: Loss convergence within 3 epochs

**Target Performance:**
- **Physics Q&A accuracy**: >0.85
- **Domain keyword coverage**: >0.90
- **Response quality**: Coherent, cited answers

## Risk Mitigation

**Training Failures:**
- Automatic checkpoint saving every 500 steps
- Resume from last checkpoint on restart
- Early stopping prevents overfitting

**Memory Issues:**
- QLoRA 4-bit quantization reduces memory by 75%
- Gradient checkpointing trades compute for memory
- Batch size auto-adjustment based on available memory

**Performance Issues:**
- Baseline evaluation establishes minimum performance
- A/B testing framework for gradual deployment
- Fallback to base model if fine-tuning fails

## Integration with ResearchMate API

After successful training, integrate with existing API:

```python
# Update backend/models/llm_manager.py
from peft import PeftModel

# Load fine-tuned model
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "fine_tuned_models/ResearchMate-Mistral-7B"
model = PeftModel.from_pretrained(base_model, adapter_path)
```

## Monitoring & Logging

**Training Metrics:**
- Loss convergence tracking
- GPU utilization monitoring
- Memory usage alerts
- Training time estimation

**Evaluation Metrics:**
- Physics domain accuracy
- Response length & quality
- Citation accuracy
- Hallucination detection

## Expected Timeline

**Day 1**: Data preparation & baseline evaluation (automated)
**Day 2-3**: LoRA training with checkpoints (2-3 days on 8x A100)
**Day 4**: Model evaluation & integration testing
**Total**: 4 days end-to-end

## Hardware Requirements

**Minimum**: 1x NVIDIA A100 80GB
**Recommended**: 4x NVIDIA A100 80GB  
**Optimal**: 8x NVIDIA A100 80GB

**Storage**: 200GB for datasets + checkpoints (not tracked in git)

## Final Deliverables

1. **LoRA Adapters**: `adapter_model.safetensors` (~20MB) - **tracked in git**
2. **Training Metrics**: Performance tracking & comparison - **tracked in git**
3. **Evaluation Report**: Detailed improvement analysis - **tracked in git**
4. **Integration Code**: Updated LLM manager for production - **tracked in git**

**Note**: Large training data (multi-GB) and intermediate checkpoints are excluded from git tracking to keep the repository lightweight. Only the final deliverable LoRA adapters (~20MB) are version controlled.

**Ready for Phase 4 deployment** with comprehensive monitoring and fallback strategies.