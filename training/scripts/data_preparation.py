#!/usr/bin/env python3
"""
Data preparation script for ResearchMate fine-tuning
Processes Physics Stack Exchange and ArXiv data into training format
"""
import os
import sys
import json
import requests
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import (
    load_config, format_physics_qa, format_arxiv_qa,
    apply_quality_filters, format_for_training,
    save_dataset, split_dataset, get_dataset_stats
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicsStackExchangeProcessor:
    """Process Physics Stack Exchange data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_data_dir = Path(config['paths']['raw_data'])
        self.processed_data_dir = Path(config['paths']['processed_data'])
        
    def download_stackexchange_data(self):
        """Download Physics StackExchange dump (simulated - actual implementation would download)"""
        logger.info("Physics StackExchange data download would happen here")
        logger.info("For now, assuming data is manually placed in training/data/raw/")
        
        # Create placeholder structure for development
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data for testing
        sample_data = [
            {
                "Id": "1",
                "Title": "What is quantum entanglement?",
                "Body": "Can someone explain quantum entanglement in simple terms?",
                "AcceptedAnswerId": "2",
                "Score": 15
            },
            {
                "Id": "2",
                "PostTypeId": "2",
                "ParentId": "1",
                "Body": "Quantum entanglement is a quantum mechanical phenomenon in which particles become correlated such that the quantum state of each particle cannot be described independently.",
                "Score": 20
            }
        ]
        
        sample_file = self.raw_data_dir / "sample_physics_qa.json"
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample data created at {sample_file}")
    
    def process_posts(self) -> List[Dict[str, str]]:
        """Process Physics StackExchange posts into Q&A format"""
        
        logger.info("Processing Physics StackExchange posts...")
        
        # Load sample data (in real implementation, this would parse XML dump)
        sample_file = self.raw_data_dir / "sample_physics_qa.json"
        
        if not sample_file.exists():
            logger.warning("No StackExchange data found. Creating sample data...")
            self.download_stackexchange_data()
        
        with open(sample_file, 'r') as f:
            posts = json.load(f)
        
        # Process into Q&A format
        qa_pairs = []
        questions = {p['Id']: p for p in posts if p.get('PostTypeId') != '2'}
        answers = {p['ParentId']: p for p in posts if p.get('PostTypeId') == '2'}
        
        for qid, question in questions.items():
            if qid in answers:
                answer = answers[qid]
                
                # Apply score filters
                min_score = self.config['data']['physics_stackexchange']['min_score']
                if question.get('Score', 0) >= min_score and answer.get('Score', 0) >= min_score:
                    
                    qa_pair = format_physics_qa(
                        question=question['Body'],
                        answer=answer['Body'],
                        title=question.get('Title')
                    )
                    qa_pairs.append(qa_pair)
        
        logger.info(f"Processed {len(qa_pairs)} Q&A pairs from StackExchange")
        return qa_pairs

class ArXivProcessor:
    """Process ArXiv physics papers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_data_dir = Path(config['paths']['raw_data'])
        
    def fetch_arxiv_papers(self) -> List[Dict[str, str]]:
        """Fetch ArXiv physics papers (simulated for development)"""
        
        logger.info("Fetching ArXiv physics papers...")
        
        # Create sample ArXiv data for testing
        sample_papers = [
            {
                "title": "Topological Insulators and Superconductors",
                "abstract": "Topological insulators are electronic materials that have a bulk band gap like an ordinary insulator but have protected conducting states on their edge or surface. These states are protected by time-reversal symmetry."
            },
            {
                "title": "Quantum Computing with Trapped Ions",
                "abstract": "Trapped atomic ions are among the most promising candidates for quantum information processing. We review the basic physics of ion trapping and discuss quantum gate operations."
            },
            {
                "title": "Many-Body Localization in Disordered Systems",
                "abstract": "Many-body localization represents a novel paradigm of ergodicity breaking in isolated quantum systems. We discuss the theoretical framework and experimental signatures."
            }
        ]
        
        # Save sample data
        sample_file = self.raw_data_dir / "sample_arxiv_papers.json"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        with open(sample_file, 'w') as f:
            json.dump(sample_papers, f, indent=2)
        
        logger.info(f"Sample ArXiv data created with {len(sample_papers)} papers")
        return sample_papers
    
    def process_papers(self) -> List[Dict[str, str]]:
        """Process ArXiv papers into Q&A format"""
        
        papers = self.fetch_arxiv_papers()
        qa_pairs = []
        
        for paper in papers:
            qa_pair = format_arxiv_qa(
                abstract=paper['abstract'],
                title=paper['title']
            )
            qa_pairs.append(qa_pair)
        
        logger.info(f"Processed {len(qa_pairs)} Q&A pairs from ArXiv")
        return qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for ResearchMate")
    parser.add_argument("--config", default="training/configs/data_config.yaml", 
                       help="Path to data configuration file")
    parser.add_argument("--output-dir", default="training/data/processed",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize processors
    stackexchange_processor = PhysicsStackExchangeProcessor(config)
    arxiv_processor = ArXivProcessor(config)
    
    # Process data sources
    logger.info("Starting data preparation...")
    
    # Process StackExchange data
    stackexchange_data = stackexchange_processor.process_posts()
    
    # Process ArXiv data
    arxiv_data = arxiv_processor.process_papers()
    
    # Combine datasets
    all_data = stackexchange_data + arxiv_data
    logger.info(f"Combined dataset size: {len(all_data)} examples")
    
    # Apply quality filters
    filtered_data = apply_quality_filters(all_data, config)
    logger.info(f"After filtering: {len(filtered_data)} examples")
    
    # Format for training
    training_data = format_for_training(filtered_data)
    
    # Split into train/validation
    train_data, val_data = split_dataset(
        training_data, 
        train_ratio=config['data']['processing']['train_split'],
        seed=config['data']['processing']['seed']
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    save_dataset(train_data, output_dir / "train_dataset.json")
    save_dataset(val_data, output_dir / "val_dataset.json")
    save_dataset(training_data, output_dir / "full_dataset.json")
    
    # Generate statistics
    train_stats = get_dataset_stats(train_data)
    val_stats = get_dataset_stats(val_data)
    
    stats = {
        "train": train_stats,
        "validation": val_stats,
        "total_samples": len(training_data),
        "processing_config": config['data']
    }
    
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Data preparation completed!")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()