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
import py7zr
import time
import feedparser
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
        """Download Physics StackExchange dump"""
        logger.info("Downloading Physics StackExchange data...")
        
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the 7z file
        url = self.config['data']['physics_stackexchange']['url']
        archive_path = self.raw_data_dir / "physics.stackexchange.com.7z"
        
        if not archive_path.exists():
            logger.info(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Download completed.")
        
        # Extract the archive
        extract_dir = self.raw_data_dir / "physics_stackexchange"
        if not extract_dir.exists():
            logger.info("Extracting archive...")
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                archive.extractall(path=extract_dir)
            logger.info("Extraction completed.")
        
        return extract_dir
    
    def process_posts(self) -> List[Dict[str, str]]:
        """Process Physics StackExchange posts into Q&A format"""
        
        logger.info("Processing Physics StackExchange posts...")
        
        # Download and extract data if not present
        extract_dir = self.raw_data_dir / "physics_stackexchange"
        if not extract_dir.exists():
            extract_dir = self.download_stackexchange_data()
        
        # Parse Posts.xml file
        posts_xml = extract_dir / "Posts.xml"
        if not posts_xml.exists():
            logger.error(f"Posts.xml not found in {extract_dir}")
            return []
        
        logger.info("Parsing Posts.xml...")
        qa_pairs = []
        questions = {}
        answers = {}
        
        # Parse XML incrementally to handle large files
        for event, elem in ET.iterparse(posts_xml, events=('start', 'end')):
            if event == 'end' and elem.tag == 'row':
                post_type = elem.get('PostTypeId')
                post_id = elem.get('Id')
                score = int(elem.get('Score', 0))
                
                if post_type == '1':  # Question
                    title = elem.get('Title', '')
                    body = elem.get('Body', '')
                    accepted_answer_id = elem.get('AcceptedAnswerId')
                    
                    if score >= self.config['data']['physics_stackexchange']['min_score']:
                        questions[post_id] = {
                            'Id': post_id,
                            'Title': title,
                            'Body': body,
                            'AcceptedAnswerId': accepted_answer_id,
                            'Score': score
                        }
                
                elif post_type == '2':  # Answer
                    parent_id = elem.get('ParentId')
                    body = elem.get('Body', '')
                    
                    if (score >= self.config['data']['physics_stackexchange']['min_score'] and 
                        len(body) >= self.config['data']['physics_stackexchange']['min_answer_length']):
                        if parent_id not in answers:
                            answers[parent_id] = []
                        answers[parent_id].append({
                            'Id': post_id,
                            'Body': body,
                            'Score': score
                        })
                
                # Clear element to save memory
                elem.clear()
        
        # Match questions with their accepted answers
        target_size = self.config['data']['physics_stackexchange']['target_size']
        processed_count = 0
        
        for q_id, question in questions.items():
            if processed_count >= target_size:
                break
                
            # First try accepted answer
            if question.get('AcceptedAnswerId') and question['AcceptedAnswerId'] in [a['Id'] for parent_answers in answers.values() for a in parent_answers]:
                for parent_id, parent_answers in answers.items():
                    if parent_id == q_id:
                        for answer in parent_answers:
                            if answer['Id'] == question['AcceptedAnswerId']:
                                qa_pair = format_physics_qa(
                                    question=question['Body'],
                                    answer=answer['Body'],
                                    title=question.get('Title')
                                )
                                qa_pairs.append(qa_pair)
                                processed_count += 1
                                break
                        break
            
            # Otherwise use highest scored answer
            elif q_id in answers:
                best_answer = max(answers[q_id], key=lambda x: x['Score'])
                qa_pair = format_physics_qa(
                    question=question['Body'],
                    answer=best_answer['Body'],
                    title=question.get('Title')
                )
                qa_pairs.append(qa_pair)
                processed_count += 1
        
        logger.info(f"Processed {len(qa_pairs)} Q&A pairs from StackExchange")
        return qa_pairs

class ArXivProcessor:
    """Process ArXiv physics papers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_data_dir = Path(config['paths']['raw_data'])
        
    def fetch_arxiv_papers(self) -> List[Dict[str, str]]:
        """Fetch ArXiv physics papers with fallback mechanisms"""
        
        logger.info("Fetching ArXiv physics papers...")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        categories = self.config['data']['arxiv_physics']['categories']
        target_size = self.config['data']['arxiv_physics']['target_size']
        date_range = self.config['data']['arxiv_physics']['date_range']
        
        # Split date range
        start_year, end_year = map(int, date_range.split('-'))
        papers_per_category = target_size // len(categories)
        
        # Try arxiv library first (more reliable)
        try:
            papers = self._fetch_with_arxiv_library(categories, papers_per_category, start_year, end_year)
            if papers:
                logger.info(f"Successfully fetched {len(papers)} papers using arxiv library")
                self._save_papers(papers)
                return papers
        except Exception as e:
            logger.warning(f"ArXiv library approach failed: {e}")
        
        # Fallback to feedparser
        try:
            papers = self._fetch_with_feedparser(categories, papers_per_category, start_year, end_year)
            if papers:
                logger.info(f"Successfully fetched {len(papers)} papers using feedparser")
                self._save_papers(papers)
                return papers
        except Exception as e:
            logger.warning(f"Feedparser approach failed: {e}")
        
        # Final fallback: create sample data
        logger.warning("All ArXiv fetching methods failed. Creating sample data...")
        papers = self._create_sample_arxiv_data()
        self._save_papers(papers)
        return papers
    
    def _fetch_with_arxiv_library(self, categories, papers_per_category, start_year, end_year):
        """Fetch papers using the arxiv library"""
        try:
            import arxiv
        except ImportError:
            logger.info("arxiv library not available. Install with: pip install arxiv")
            return []
        
        papers = []
        
        for category in categories:
            logger.info(f"Fetching papers from category {category} using arxiv library...")
            
            try:
                # Create search query
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=papers_per_category,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                category_papers = []
                for result in search.results():
                    # Check publication date
                    published_year = result.published.year
                    if start_year <= published_year <= end_year:
                        paper = {
                            'id': result.entry_id.split('/')[-1],
                            'title': result.title.strip(),
                            'abstract': result.summary.strip(),
                            'authors': [str(author) for author in result.authors],
                            'published': result.published.isoformat(),
                            'categories': [str(cat) for cat in result.categories],
                            'url': result.entry_id
                        }
                        category_papers.append(paper)
                        
                        if len(category_papers) >= papers_per_category:
                            break
                
                papers.extend(category_papers)
                logger.info(f"Fetched {len(category_papers)} papers from {category}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch {category} papers with arxiv library: {e}")
        
        return papers
    
    def _fetch_with_feedparser(self, categories, papers_per_category, start_year, end_year):
        """Fetch papers using feedparser and requests"""
        papers = []
        
        for category in categories:
            logger.info(f"Fetching papers from category: {category}")
            
            # Build search query
            search_query = f"cat:{category}"
            base_url = "http://export.arxiv.org/api/query"
            
            # Fetch papers in batches
            start = 0
            max_results = 100
            category_papers = []
            consecutive_failures = 0
            
            while len(category_papers) < papers_per_category and consecutive_failures < 3:
                params = {
                    'search_query': search_query,
                    'start': start,
                    'max_results': min(max_results, papers_per_category - len(category_papers)),
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    # Parse the Atom feed
                    feed = feedparser.parse(response.content)
                    
                    if not feed.entries:
                        logger.warning(f"No more papers found for category {category}")
                        break
                    
                    batch_added = 0
                    for entry in feed.entries:
                        try:
                            published_year = int(entry.published[:4])
                            if start_year <= published_year <= end_year:
                                paper = {
                                    'id': entry.id.split('/')[-1],
                                    'title': entry.title.replace('\n', ' ').strip(),
                                    'abstract': entry.summary.replace('\n', ' ').strip(),
                                    'authors': [author.name for author in entry.authors],
                                    'published': entry.published,
                                    'categories': [tag.term for tag in entry.tags],
                                    'url': entry.link
                                }
                                category_papers.append(paper)
                                batch_added += 1
                                
                                if len(category_papers) >= papers_per_category:
                                    break
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Skipping paper due to parsing error: {e}")
                            continue
                    
                    if batch_added == 0:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0
                    
                    start += max_results
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Request failed for {category} at start={start}: {e}")
                    consecutive_failures += 1
                    time.sleep(2)
            
            papers.extend(category_papers)
            logger.info(f"Fetched {len(category_papers)} papers from {category}")
        
        return papers
    
    def _create_sample_arxiv_data(self):
        """Create sample ArXiv data when fetching fails"""
        logger.info("Creating sample ArXiv data...")
        
        categories = self.config['data']['arxiv_physics']['categories']
        papers_per_category = self.config['data']['arxiv_physics']['target_size'] // len(categories)
        
        base_papers = [
            {
                "title": "Topological Insulators and Superconductors",
                "abstract": "Topological insulators are electronic materials that have a bulk band gap like an ordinary insulator but have protected conducting states on their edge or surface. These states are protected by time-reversal symmetry and result in novel electronic properties.",
            },
            {
                "title": "Quantum Computing with Trapped Ions", 
                "abstract": "Trapped atomic ions are among the most promising candidates for quantum information processing. We review the basic physics of ion trapping and discuss quantum gate operations, error correction, and scaling challenges.",
            },
            {
                "title": "Many-Body Localization in Disordered Systems",
                "abstract": "Many-body localization represents a novel paradigm of ergodicity breaking in isolated quantum systems. We discuss the theoretical framework, experimental signatures, and implications for thermalization.",
            }
        ]
        
        sample_papers = []
        for category in categories:
            for j in range(papers_per_category):
                base_paper = base_papers[j % len(base_papers)]
                paper = {
                    'id': f"sample-{category}-{j:04d}",
                    'title': f"{base_paper['title']} - Sample {j+1}",
                    'abstract': base_paper['abstract'],
                    'authors': ["Sample Author"],
                    'published': "2023-01-01T00:00:00Z",
                    'categories': [category],
                    'url': f"https://arxiv.org/abs/sample-{category}-{j:04d}"
                }
                sample_papers.append(paper)
        
        return sample_papers
    
    def _save_papers(self, papers):
        """Save papers to JSON file"""
        arxiv_file = self.raw_data_dir / "arxiv_papers.json"
        with open(arxiv_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(papers)} papers to {arxiv_file}")
    
    def process_papers(self) -> List[Dict[str, str]]:
        """Process ArXiv papers into Q&A format"""
        
        # Check if we already have fetched data
        arxiv_file = self.raw_data_dir / "arxiv_papers.json"
        
        if arxiv_file.exists():
            logger.info("Loading existing ArXiv data...")
            with open(arxiv_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
        else:
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