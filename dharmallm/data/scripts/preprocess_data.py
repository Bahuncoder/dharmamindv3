"""
Data Preprocessing for DharmaLLM
===============================

This module handles the preprocessing of spiritual and dharmic texts
for training the DharmaLLM model.
"""

import json
import re
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DharmaDataPreprocessor:
    """Preprocesses spiritual and dharmic texts for model training."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for training."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere with training
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\'\"]+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def create_qa_pairs(self, content: str, title: str = "") -> List[Dict[str, str]]:
        """Create question-answer pairs from spiritual content."""
        qa_pairs = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
                
            # Generate different types of questions
            questions = self.generate_questions_for_paragraph(paragraph, title)
            
            for question in questions:
                qa_pairs.append({
                    'input': question,
                    'output': paragraph,
                    'source': title,
                    'type': 'wisdom_sharing'
                })
                
        return qa_pairs
    
    def generate_questions_for_paragraph(self, paragraph: str, title: str = "") -> List[str]:
        """Generate relevant questions for a paragraph of wisdom."""
        questions = []
        
        # Analyze content for key themes
        content_lower = paragraph.lower()
        
        # Meditation-related questions
        if any(word in content_lower for word in ['meditation', 'mindfulness', 'breathing', 'awareness']):
            questions.extend([
                "How can I practice meditation?",
                "What is mindfulness?",
                "Can you guide me in meditation?",
                "How do I become more aware?",
                "What are meditation techniques?"
            ])
            
        # Ethics and dharma questions
        if any(word in content_lower for word in ['dharma', 'ethics', 'right', 'moral', 'virtue']):
            questions.extend([
                "What is dharma?",
                "How should I live ethically?",
                "What is right action?",
                "How do I make moral decisions?",
                "What are spiritual virtues?"
            ])
            
        # Suffering and healing questions
        if any(word in content_lower for word in ['suffering', 'pain', 'grief', 'healing', 'peace']):
            questions.extend([
                "How do I deal with suffering?",
                "Why do we experience pain?",
                "How can I find peace?",
                "How do I heal from grief?",
                "What helps with emotional pain?"
            ])
            
        # Relationship questions
        if any(word in content_lower for word in ['love', 'relationship', 'compassion', 'kindness']):
            questions.extend([
                "How do I show love?",
                "What makes relationships work?",
                "How can I be more compassionate?",
                "What is true love?",
                "How do I practice kindness?"
            ])
            
        # Purpose and meaning questions
        if any(word in content_lower for word in ['purpose', 'meaning', 'life', 'existence']):
            questions.extend([
                "What is the meaning of life?",
                "How do I find my purpose?",
                "Why do we exist?",
                "What is life about?",
                "How do I find direction?"
            ])
            
        # If no specific themes found, use general questions
        if not questions:
            questions = [
                "Can you share some wisdom?",
                "What guidance do you have?",
                "How can I grow spiritually?",
                "What should I understand about life?",
                "Share some spiritual insight."
            ]
            
        return questions[:3]  # Limit to 3 questions per paragraph
    
    def process_scripture_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a scripture or spiritual text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Clean the content
            content = self.clean_text(content)
            
            # Create QA pairs
            qa_pairs = self.create_qa_pairs(content, file_path.stem)
            
            logger.info(f"Processed {file_path.name}: {len(qa_pairs)} QA pairs created")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def create_sample_data(self):
        """Create sample training data for demonstration."""
        sample_data = [
            {
                'input': 'How can I find inner peace?',
                'output': 'Inner peace comes from accepting what is beyond your control and taking mindful action on what you can influence. Practice daily meditation, cultivate gratitude, and remember that peace is not the absence of chaos, but the presence of calm within it. Start each day with intention and end it with reflection.',
                'wisdom_category': 'peace',
                'dharmic_score': 0.9
            },
            {
                'input': 'What is the meaning of suffering?',
                'output': 'Suffering is often our greatest teacher. It arises from attachment to how we think things should be, rather than accepting how they are. The Buddha taught that suffering comes from craving and aversion. When we learn to observe our pain without identifying with it, we find freedom. Suffering can crack us open to deeper compassion and wisdom.',
                'wisdom_category': 'suffering',
                'dharmic_score': 0.95
            },
            {
                'input': 'How do I practice loving-kindness?',
                'output': 'Begin with yourself - offer yourself the same kindness you would give a dear friend. Then extend this loving energy to loved ones, neutral people, difficult people, and finally all beings. Use phrases like "May you be happy, may you be healthy, may you be at peace." Practice daily, even for just five minutes. Remember, love is both a feeling and an action.',
                'wisdom_category': 'compassion',
                'dharmic_score': 0.92
            },
            {
                'input': 'What is dharma?',
                'output': 'Dharma is your unique path of righteous living - the way you can serve the world with your particular gifts and circumstances. It encompasses both universal principles of ethics and your personal spiritual duty. When you align with dharma, actions flow naturally from wisdom and compassion. It is both the path and the destination of a meaningful life.',
                'wisdom_category': 'dharma',
                'dharmic_score': 0.98
            },
            {
                'input': 'How do I meditate as a beginner?',
                'output': 'Start simple: sit comfortably, close your eyes, and focus on your natural breath. When thoughts arise - and they will - gently notice them and return to your breath. Begin with just 5-10 minutes daily. There is no "perfect" meditation, only honest practice. The goal is not to stop thoughts but to develop a peaceful relationship with them.',
                'wisdom_category': 'meditation',
                'dharmic_score': 0.88
            }
        ]
        
        # Save training data
        train_file = self.processed_data_dir / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
                
        # Create smaller eval set
        eval_data = sample_data[:2]  # Use first 2 items for eval
        eval_file = self.processed_data_dir / "eval.jsonl"
        with open(eval_file, 'w', encoding='utf-8') as f:
            for item in eval_data:
                f.write(json.dumps(item) + '\n')
                
        logger.info(f"Created sample data: {len(sample_data)} training samples, {len(eval_data)} eval samples")
    
    def process_all_files(self):
        """Process all files in the raw data directory."""
        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data directory {self.raw_data_dir} does not exist. Creating sample data instead.")
            self.create_sample_data()
            return
            
        all_qa_pairs = []
        
        # Process text files
        for file_path in self.raw_data_dir.glob("*.txt"):
            qa_pairs = self.process_scripture_file(file_path)
            all_qa_pairs.extend(qa_pairs)
            
        if not all_qa_pairs:
            logger.warning("No data files found. Creating sample data.")
            self.create_sample_data()
            return
            
        # Split into train/eval
        split_point = int(len(all_qa_pairs) * 0.9)
        train_data = all_qa_pairs[:split_point]
        eval_data = all_qa_pairs[split_point:]
        
        # Save processed data
        self.save_data(train_data, "train.jsonl")
        self.save_data(eval_data, "eval.jsonl")
        
        logger.info(f"Processed {len(all_qa_pairs)} total QA pairs")
        logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    def save_data(self, data: List[Dict[str, str]], filename: str):
        """Save data to JSONL format."""
        file_path = self.processed_data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

def main():
    """Main preprocessing script."""
    preprocessor = DharmaDataPreprocessor()
    preprocessor.process_all_files()

if __name__ == "__main__":
    main()
