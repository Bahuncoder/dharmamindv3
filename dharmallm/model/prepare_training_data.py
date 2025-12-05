"""
🕉️ DharmaLLM Data Preparation
============================

Consolidate and prepare all dharmic training data for the custom model.
Creates a clean, deduplicated, and properly formatted training corpus.

Usage:
    python prepare_training_data.py
    python prepare_training_data.py --output data/processed_corpus.txt

May this data preparation honor the sacred texts! 🙏
"""

import os
import sys
import json
import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DharmicDataPreparer:
    """Prepare and consolidate dharmic training data."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.texts: List[str] = []
        self.text_hashes: Set[str] = set()
        self.stats = defaultdict(int)
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        # Normalize and hash
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized[:500].encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters (except newlines)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1{2,}', r'\1', text)
        
        return text.strip()
    
    def _is_quality_text(self, text: str) -> bool:
        """Check if text is quality content."""
        # Minimum length
        if len(text) < 50:
            return False
        
        # Not just numbers/symbols
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return False
        
        # Not excessive repetition
        words = text.split()
        if len(words) < 5:
            return False
        
        unique_words = len(set(words))
        if unique_words / len(words) < 0.3:  # Too repetitive
            return False
        
        return True
    
    def add_text(self, text: str, source: str = "unknown"):
        """Add text if it's quality and not duplicate."""
        text = self._clean_text(text)
        
        if not self._is_quality_text(text):
            self.stats['filtered'] += 1
            return False
        
        text_hash = self._compute_hash(text)
        if text_hash in self.text_hashes:
            self.stats['duplicates'] += 1
            return False
        
        self.text_hashes.add(text_hash)
        self.texts.append(text)
        self.stats[source] += 1
        self.stats['total'] += 1
        return True
    
    def process_txt_file(self, file_path: Path):
        """Process a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', content)
            
            for para in paragraphs:
                para = para.strip()
                if para:
                    self.add_text(para, 'txt_files')
                    
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
    
    def process_json_file(self, file_path: Path):
        """Process a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            texts = self._extract_texts_from_json(data)
            for text in texts:
                self.add_text(text, 'json_files')
                
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
    
    def _extract_texts_from_json(self, data, depth: int = 0) -> List[str]:
        """Recursively extract text from JSON."""
        texts = []
        
        if depth > 10:
            return texts
        
        if isinstance(data, str):
            if len(data) >= 50:
                texts.append(data)
        elif isinstance(data, list):
            for item in data:
                texts.extend(self._extract_texts_from_json(item, depth + 1))
        elif isinstance(data, dict):
            # Priority text fields
            text_fields = [
                'text', 'content', 'translation', 'meaning', 'explanation',
                'verse', 'shloka', 'wisdom', 'teaching', 'commentary',
                'sanskrit', 'english', 'description', 'message', 'guidance',
                'answer', 'response', 'body', 'summary'
            ]
            
            for field in text_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, str) and len(value) >= 50:
                        texts.append(value)
            
            # Recurse into other fields
            for key, value in data.items():
                if key not in text_fields and isinstance(value, (list, dict)):
                    texts.extend(self._extract_texts_from_json(value, depth + 1))
        
        return texts
    
    def process_jsonl_file(self, file_path: Path):
        """Process a JSONL file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        texts = self._extract_texts_from_json(item)
                        for text in texts:
                            self.add_text(text, 'jsonl_files')
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
    
    def process_all(self):
        """Process all data files."""
        logger.info(f"Processing data from: {self.data_dir}")
        
        # Process text files
        for file_path in self.data_dir.glob('**/*.txt'):
            self.process_txt_file(file_path)
        
        # Process JSON files
        for file_path in self.data_dir.glob('**/*.json'):
            self.process_json_file(file_path)
        
        # Process JSONL files
        for file_path in self.data_dir.glob('**/*.jsonl'):
            self.process_jsonl_file(file_path)
        
        logger.info(f"Processing complete!")
        logger.info(f"Statistics: {dict(self.stats)}")
    
    def add_dharmic_prompt_templates(self):
        """Add prompt-response templates for instruction tuning."""
        templates = [
            # Karma
            ("What is karma?", 
             "Karma is the universal law of cause and effect. Every action, thought, and intention creates energy that returns to us. Good actions create positive karma, while harmful actions create negative karma. The Bhagavad Gita teaches us to perform our duties without attachment to results, thus breaking the cycle of karma."),
            
            # Dharma
            ("What is dharma?",
             "Dharma is the cosmic order and righteous path that sustains the universe. It encompasses our moral duty, ethical living, and spiritual purpose. Each person has their own svadharma - their unique purpose aligned with their nature and stage of life. Following dharma brings harmony to self and society."),
            
            # Moksha
            ("What is moksha?",
             "Moksha is liberation from the cycle of birth and death (samsara). It is the ultimate goal of spiritual life - the realization of our true nature as Atman (soul) and its unity with Brahman (supreme consciousness). Moksha is achieved through knowledge (jnana), devotion (bhakti), selfless action (karma yoga), and meditation."),
            
            # Meditation
            ("How do I meditate?",
             "Begin by finding a quiet place and sitting comfortably with spine erect. Close your eyes and focus on your breath. As thoughts arise, observe them without judgment and gently return to the breath. Start with 10-15 minutes daily. Regular practice calms the mind and reveals the inner Self. Om is often used as a focal point for meditation."),
            
            # Gita wisdom
            ("What does the Bhagavad Gita teach?",
             "The Bhagavad Gita teaches the path to righteous living and spiritual liberation. Lord Krishna guides Arjuna through three main paths: Karma Yoga (selfless action), Bhakti Yoga (devotion), and Jnana Yoga (knowledge). The central teaching is to perform one's duty without attachment to results, dedicating all actions to the Divine."),
            
            # Yoga
            ("What is the purpose of yoga?",
             "Yoga means 'union' - the union of individual consciousness with universal consciousness. Beyond physical postures (asanas), yoga encompasses ethical living (yamas/niyamas), breath control (pranayama), sense withdrawal (pratyahara), concentration (dharana), meditation (dhyana), and ultimately samadhi (enlightenment)."),
            
            # Atman
            ("What is Atman?",
             "Atman is the eternal Self, the innermost essence of every being. It is distinct from the body, mind, and ego. The Upanishads teach that Atman is identical with Brahman - the universal consciousness. 'Tat Tvam Asi' - You are That. Realizing this truth leads to liberation."),
            
            # Suffering
            ("How can I overcome suffering?",
             "Suffering arises from attachment, desire, and ignorance of our true nature. The path to peace involves: 1) Practicing detachment (vairagya) while remaining engaged in life, 2) Cultivating wisdom through study and meditation, 3) Surrendering the ego to the Divine, 4) Serving others selflessly. Remember - you are not the body or mind, but the eternal witness."),
            
            # Peace
            ("How can I find inner peace?",
             "Inner peace comes from within, not from external circumstances. Practice these: Daily meditation to calm the mind. Contentment (santosha) with what you have. Non-attachment to outcomes. Service to others. Regular study of sacred texts. Spending time in nature. Remember that the peace you seek is already within you as your true nature."),
            
            # Vedas
            ("What are the Vedas?",
             "The Vedas are the oldest and most sacred texts of Sanatan Dharma, revealed to ancient Rishis in deep meditation. There are four Vedas: Rigveda (hymns), Yajurveda (rituals), Samaveda (melodies), and Atharvaveda (practical wisdom). They contain mantras, philosophy, and guidance for all aspects of life."),
            
            # Upanishads  
            ("What do the Upanishads teach?",
             "The Upanishads are the philosophical crown of the Vedas, teaching the nature of ultimate reality. Key teachings include: Brahman is the sole reality, Atman and Brahman are one, the world is Maya (illusion), and liberation comes through knowledge. Famous declarations include 'Aham Brahmasmi' (I am Brahman) and 'Tat Tvam Asi' (You are That)."),
        ]
        
        for question, answer in templates:
            # Add as Q&A format
            qa_text = f"Question: {question}\n\nAnswer: {answer}"
            self.add_text(qa_text, 'templates')
            
            # Also add just the answer as standalone wisdom
            self.add_text(answer, 'templates')
        
        logger.info(f"Added {len(templates)} dharmic templates")
    
    def save(self, output_path: Path, format: str = 'txt'):
        """Save processed data."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in self.texts:
                    f.write(text + '\n\n')
        
        elif format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in self.texts:
                    json.dump({'text': text}, f, ensure_ascii=False)
                    f.write('\n')
        
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.texts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.texts)} texts to {output_path}")
        
        # Save stats
        stats_path = output_path.with_suffix('.stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'total_texts': len(self.texts),
                'total_chars': sum(len(t) for t in self.texts),
                'sources': dict(self.stats)
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare DharmaLLM training data')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='data/processed/training_corpus.txt',
                        help='Output file path')
    parser.add_argument('--format', choices=['txt', 'jsonl', 'json'], default='txt',
                        help='Output format')
    parser.add_argument('--add_templates', action='store_true', default=True,
                        help='Add dharmic Q&A templates')
    
    args = parser.parse_args()
    
    logger.info("🕉️ DharmaLLM Data Preparation")
    logger.info("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_path = base_dir / args.output
    
    preparer = DharmicDataPreparer(data_dir)
    preparer.process_all()
    
    if args.add_templates:
        preparer.add_dharmic_prompt_templates()
    
    preparer.save(output_path, args.format)
    
    logger.info(f"\n✅ Data preparation complete!")
    logger.info(f"   Total texts: {len(preparer.texts)}")
    logger.info(f"   Total chars: {sum(len(t) for t in preparer.texts):,}")
    logger.info(f"   Output: {output_path}")


if __name__ == '__main__':
    main()

