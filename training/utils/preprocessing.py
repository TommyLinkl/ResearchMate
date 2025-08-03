"""
Data preprocessing utilities for ResearchMate fine-tuning
"""
import json
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_html(text: str) -> str:
    """Remove HTML tags and clean text"""
    if not text:
        return ""
    
    # Parse HTML and extract text
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def format_physics_qa(question: str, answer: str, title: str = None) -> Dict[str, str]:
    """Format physics Q&A into instruction-following format"""
    
    # Clean inputs
    question = clean_html(question)
    answer = clean_html(answer)
    
    if title:
        title = clean_html(title)
        instruction = f"Answer this physics question: {title}\n\n{question}"
    else:
        instruction = f"Answer this physics question: {question}"
    
    # Format response with proper structure
    formatted_answer = f"{answer}"
    
    return {
        "instruction": instruction.strip(),
        "input": "",
        "output": formatted_answer.strip()
    }

def format_arxiv_qa(abstract: str, title: str) -> Dict[str, str]:
    """Convert ArXiv abstract into Q&A format"""
    
    title = clean_html(title)
    abstract = clean_html(abstract)
    
    # Generate question from title
    instruction = f"Explain the key concepts and findings from this physics research: {title}"
    
    # Format abstract as explanation
    output = f"This research focuses on {abstract}"
    
    return {
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip()
    }

def apply_quality_filters(data: List[Dict[str, str]], config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Apply quality filters to training data"""
    
    filters = config['data']['filters']
    filtered_data = []
    
    for item in data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # Length filters
        if len(instruction) < filters['min_instruction_length']:
            continue
        if len(instruction) > filters['max_instruction_length']:
            continue
        if len(output) < filters['min_output_length']:
            continue
        if len(output) > filters['max_output_length']:
            continue
            
        # Quality filters
        if not instruction.strip() or not output.strip():
            continue
            
        filtered_data.append(item)
    
    return filtered_data

def format_for_training(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format data for instruction tuning"""
    
    formatted_data = []
    
    for item in data:
        # Create conversational format
        if item['input'].strip():
            text = f"<s>[INST] {item['instruction']}\n\n{item['input']} [/INST] {item['output']}</s>"
        else:
            text = f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"
            
        formatted_data.append({
            "text": text,
            "instruction": item['instruction'],
            "input": item['input'],
            "output": item['output']
        })
    
    return formatted_data

def save_dataset(data: List[Dict[str, str]], filepath: str):
    """Save dataset to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_dataset(filepath: str) -> List[Dict[str, str]]:
    """Load dataset from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_dataset(data: List[Dict[str, str]], train_ratio: float = 0.9, seed: int = 42) -> tuple:
    """Split dataset into train and validation sets"""
    import random
    
    random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def get_dataset_stats(data: List[Dict[str, str]]) -> Dict[str, Any]:
    """Calculate dataset statistics"""
    
    if not data:
        return {"total_samples": 0}
    
    instruction_lengths = [len(item['instruction']) for item in data]
    output_lengths = [len(item['output']) for item in data]
    
    stats = {
        "total_samples": len(data),
        "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths),
        "avg_output_length": sum(output_lengths) / len(output_lengths),
        "max_instruction_length": max(instruction_lengths),
        "max_output_length": max(output_lengths),
        "min_instruction_length": min(instruction_lengths),
        "min_output_length": min(output_lengths)
    }
    
    return stats