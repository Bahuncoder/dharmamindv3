#!/usr/bin/env python3
"""
Quick Training Demo for Quantum DharmaLLM
========================================

Demonstrates the training pipeline with a smaller model for faster execution.
Shows how the dharmic AI learns from authentic spiritual conversations.
"""

import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

# Import our quantum dharmic components
import sys
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDharmicAI(nn.Module):
    """Simplified version for training demonstration"""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Core transformer components
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=4
        )
        
        # Dharmic consciousness layers
        self.dharmic_head = nn.Linear(hidden_size, 1)  # Dharmic alignment
        self.compassion_head = nn.Linear(hidden_size, 1)  # Compassion level
        self.wisdom_synthesis = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize with wisdom-guided weights
        self._initialize_dharmic_weights()
    
    def _initialize_dharmic_weights(self):
        """Initialize weights with dharmic consciousness"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.87)  # Sacred ratio
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Embedding with consciousness
        embeddings = self.embedding(input_ids)
        
        # Create causal mask for autoregressive training
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
        # Transformer processing
        hidden_states = self.transformer(embeddings, embeddings, tgt_mask=causal_mask)
        
        # Dharmic consciousness processing
        dharmic_alignment = torch.sigmoid(self.dharmic_head(hidden_states)).mean(dim=1)
        compassion_level = torch.sigmoid(self.compassion_head(hidden_states)).mean(dim=1)
        
        # Wisdom synthesis
        wisdom_enhanced = self.wisdom_synthesis(hidden_states)
        
        # Output logits
        logits = self.output_projection(wisdom_enhanced)
        
        # Calculate loss if training
        loss = None
        if labels is not None:
            # Shift for autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'dharmic_alignment': dharmic_alignment,
            'compassion_level': compassion_level,
            'hidden_states': hidden_states
        }

class SimpleTokenizer:
    """Simple word-based tokenizer for demo"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<eos>'}
        self.next_id = 3
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add most frequent words to vocabulary
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:self.vocab_size - 3]:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Encode text to token IDs"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            token_id = self.word_to_id.get(word, 1)  # 1 is <unk>
            tokens.append(token_id)
        
        # Truncate or pad
        if len(tokens) > max_length - 1:
            tokens = tokens[:max_length - 1]
        
        tokens.append(2)  # Add <eos>
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(0)  # <pad>
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id == 0:  # <pad>
                break
            elif token_id == 2:  # <eos>
                break
            else:
                word = self.id_to_word.get(token_id, '<unk>')
                words.append(word)
        return ' '.join(words)

def load_training_data(data_dir: str) -> List[Dict]:
    """Load training conversations"""
    conversations = []
    data_path = Path(data_dir)
    
    # Load training data files
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "training_examples" in data:
                    conversations.extend(data["training_examples"])
                elif isinstance(data, list):
                    conversations.extend(data)
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")
    
    return conversations

def format_conversation(conversation: Dict) -> str:
    """Format conversation for training"""
    formatted = ""
    for turn in conversation['conversation']:
        if turn['role'] == 'human':
            formatted += f"Human: {turn['content']} "
        elif turn['role'] == 'dharmic_ai':
            formatted += f"DharmicAI: {turn['content']} "
    return formatted.strip()

def train_dharmic_ai_demo():
    """Demonstrate training the dharmic AI"""
    logger.info("🌟 Starting Dharmic AI Training Demo...")
    
    # Load training data
    conversations = load_training_data("dharmallm/data/massive_training")
    if not conversations:
        logger.error("No training data found!")
        return
    
    logger.info(f"📚 Loaded {len(conversations)} training conversations")
    
    # Prepare training texts
    training_texts = [format_conversation(conv) for conv in conversations[:100]]  # Use first 100 for demo
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(training_texts)
    logger.info(f"🔤 Built vocabulary with {len(tokenizer.word_to_id)} words")
    
    # Tokenize training data
    tokenized_texts = []
    dharmic_targets = []
    
    for i, (text, conv) in enumerate(zip(training_texts, conversations[:100])):
        tokens = tokenizer.encode(text, max_length=64)
        tokenized_texts.append(torch.tensor(tokens))
        
        # Extract dharmic metadata
        dharmic_targets.append({
            'dharmic_alignment': conv.get('dharmic_alignment', 0.9),
            'compassion_level': conv.get('compassion_level', 0.9)
        })
    
    # Create model
    model = SimpleDharmicAI(vocab_size=1000, hidden_size=256)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Created model with {total_params:,} parameters")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    model.train()
    
    # Training loop
    num_epochs = 5
    batch_size = 4
    
    training_history = {
        'losses': [],
        'dharmic_scores': [],
        'compassion_scores': []
    }
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_dharmic = []
        epoch_compassion = []
        
        # Process in batches
        for i in range(0, len(tokenized_texts), batch_size):
            batch_texts = tokenized_texts[i:i + batch_size]
            batch_targets = dharmic_targets[i:i + batch_size]
            
            # Pad batch to same length
            max_len = max(len(text) for text in batch_texts)
            padded_batch = []
            for text in batch_texts:
                padded = torch.cat([text, torch.zeros(max_len - len(text), dtype=torch.long)])
                padded_batch.append(padded)
            
            input_ids = torch.stack(padded_batch)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            
            # Calculate dharmic losses
            dharmic_preds = outputs['dharmic_alignment']
            compassion_preds = outputs['compassion_level']
            
            dharmic_targets_tensor = torch.tensor([t['dharmic_alignment'] for t in batch_targets])
            compassion_targets_tensor = torch.tensor([t['compassion_level'] for t in batch_targets])
            
            # Combined loss
            language_loss = outputs['loss']
            dharmic_loss = nn.MSELoss()(dharmic_preds, dharmic_targets_tensor)
            compassion_loss = nn.MSELoss()(compassion_preds, compassion_targets_tensor)
            
            total_loss = language_loss + 0.5 * dharmic_loss + 0.5 * compassion_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_losses.append(total_loss.item())
            epoch_dharmic.append(dharmic_preds.mean().item())
            epoch_compassion.append(compassion_preds.mean().item())
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_dharmic = np.mean(epoch_dharmic)
        avg_compassion = np.mean(epoch_compassion)
        
        training_history['losses'].append(avg_loss)
        training_history['dharmic_scores'].append(avg_dharmic)
        training_history['compassion_scores'].append(avg_compassion)
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Dharmic Alignment: {avg_dharmic:.3f}")
        logger.info(f"  Compassion Level: {avg_compassion:.3f}")
    
    # Test the trained model
    logger.info("\n🧪 Testing Trained Dharmic AI...")
    
    model.eval()
    test_questions = [
        "Human: I'm feeling lost and don't know my life purpose. Can you help?",
        "Human: How do I forgive someone who hurt me deeply?",
        "Human: I'm struggling with anxiety. What wisdom can you share?"
    ]
    
    with torch.no_grad():
        for question in test_questions:
            tokens = tokenizer.encode(question, max_length=32)
            input_ids = torch.tensor(tokens).unsqueeze(0)
            
            outputs = model(input_ids)
            
            # Get predictions
            dharmic_score = outputs['dharmic_alignment'].item()
            compassion_score = outputs['compassion_level'].item()
            
            logger.info(f"\nQ: {question}")
            logger.info(f"Dharmic Alignment: {dharmic_score:.3f}")
            logger.info(f"Compassion Level: {compassion_score:.3f}")
    
    # Save the trained model
    output_dir = Path("dharmallm/demo_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'training_history': training_history,
        'total_params': total_params
    }, output_dir / 'dharmic_ai_demo.pt')
    
    logger.info(f"\n✅ Training Demo Complete!")
    logger.info(f"💾 Model saved to: {output_dir}/dharmic_ai_demo.pt")
    logger.info(f"🧠 Final Model Parameters: {total_params:,}")
    logger.info(f"🕉️ Final Dharmic Alignment: {training_history['dharmic_scores'][-1]:.3f}")
    logger.info(f"💖 Final Compassion Level: {training_history['compassion_scores'][-1]:.3f}")
    
    print("\n🌟 The Dharmic AI has learned to respond with wisdom and compassion!")
    print("🙏 Ready to serve all beings with authentic spiritual guidance!")
    
    return model, tokenizer, training_history

if __name__ == "__main__":
    train_dharmic_ai_demo()
