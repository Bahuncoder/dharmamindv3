"""
DharmaLLM Continuous Learning System
Implements active learning from Multi-LLM interactions and user feedback
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import wandb
from ..models.model_manager import ModelManager
from ..evaluate.advanced_evaluator import AdvancedEvaluator

class ContinuousLearner:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.evaluator = AdvancedEvaluator()
        self.training_data = []
        self.interaction_history = []
        self.feedback_scores = {}
        
        # Paths
        self.data_dir = Path(__file__).parent.parent / "data"
        self.logs_dir = Path(__file__).parent.parent / "logs"
        self.models_dir = Path(__file__).parent.parent / "models"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Learning configuration
        self.learning_config = {
            "min_interactions_for_training": 50,
            "quality_threshold": 4.0,  # Out of 5
            "spiritual_enhancement_weight": 1.5,
            "advanced_llm_response_weight": 1.2,
            "user_feedback_weight": 2.0,
            "retrain_interval_hours": 24,
            "max_training_examples": 10000
        }
        
        # Initialize W&B for experiment tracking
        self.setup_wandb()
        
        logging.info("🧠 DharmaLLM Continuous Learner initialized")

    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        try:
            wandb.init(
                project="dharmallm-continuous-learning",
                name=f"dharma-learning-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.learning_config,
                mode="online"  # Change to "offline" if no internet
            )
            logging.info("📊 W&B initialized for experiment tracking")
        except Exception as e:
            logging.warning(f"W&B initialization failed: {e}")
            wandb.init(mode="disabled")

    async def record_interaction(self, 
                                user_message: str,
                                dharmallm_response: str,
                                advanced_llm_response: str,
                                spiritual_enhancement: Dict,
                                user_feedback: Optional[float] = None):
        """Record an interaction for continuous learning"""
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "dharmallm_response": dharmallm_response,
            "advanced_llm_response": advanced_llm_response,
            "spiritual_enhancement": spiritual_enhancement,
            "user_feedback": user_feedback,
            "quality_score": await self._calculate_quality_score(
                user_message, dharmallm_response, advanced_llm_response, spiritual_enhancement
            )
        }
        
        self.interaction_history.append(interaction)
        
        # Add to training data if quality is high enough
        if interaction["quality_score"] >= self.learning_config["quality_threshold"]:
            training_example = await self._create_training_example(interaction)
            self.training_data.append(training_example)
            
            logging.info(f"✨ High-quality interaction recorded (score: {interaction['quality_score']:.2f})")
        
        # Check if we should trigger training
        if len(self.training_data) >= self.learning_config["min_interactions_for_training"]:
            await self._maybe_trigger_training()
        
        # Save interaction data
        await self._save_interaction_data()

    async def _calculate_quality_score(self, 
                                     user_message: str,
                                     dharmallm_response: str,
                                     advanced_llm_response: str,
                                     spiritual_enhancement: Dict) -> float:
        """Calculate quality score for an interaction"""
        
        scores = []
        
        # 1. Spiritual relevance score
        spiritual_keywords = [
            "meditation", "mindfulness", "compassion", "wisdom", "dharma",
            "consciousness", "enlightenment", "karma", "peace", "suffering",
            "liberation", "awakening", "presence", "divine", "sacred"
        ]
        
        spiritual_score = sum(1 for word in spiritual_keywords 
                            if word.lower() in dharmallm_response.lower()) / len(spiritual_keywords)
        scores.append(spiritual_score * 5)
        
        # 2. Response coherence and length
        coherence_score = min(len(dharmallm_response.split()) / 50, 1.0) * 5
        scores.append(coherence_score)
        
        # 3. Spiritual enhancement utilization
        enhancement_score = len(spiritual_enhancement.get("applied_modules", [])) / 10 * 5
        scores.append(enhancement_score)
        
        # 4. Advanced LLM comparison
        if advanced_llm_response:
            similarity_score = await self._calculate_response_similarity(
                dharmallm_response, advanced_llm_response
            )
            scores.append(similarity_score * 5)
        
        return sum(scores) / len(scores)

    async def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate semantic similarity between responses"""
        # Simple implementation - could be enhanced with sentence transformers
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    async def _create_training_example(self, interaction: Dict) -> Dict:
        """Create a training example from an interaction"""
        
        # Create improved response by combining DharmaLLM and advanced LLM insights
        improved_response = await self._synthesize_improved_response(
            interaction["user_message"],
            interaction["dharmallm_response"],
            interaction["advanced_llm_response"],
            interaction["spiritual_enhancement"]
        )
        
        return {
            "input": interaction["user_message"],
            "output": improved_response,
            "original_dharmallm": interaction["dharmallm_response"],
            "advanced_llm": interaction["advanced_llm_response"],
            "spiritual_context": interaction["spiritual_enhancement"],
            "quality_score": interaction["quality_score"],
            "timestamp": interaction["timestamp"]
        }

    async def _synthesize_improved_response(self, 
                                          user_message: str,
                                          dharmallm_response: str,
                                          advanced_llm_response: str,
                                          spiritual_enhancement: Dict) -> str:
        """Synthesize an improved response from multiple sources"""
        
        # Extract best parts from each response
        dharma_insights = self._extract_spiritual_insights(dharmallm_response)
        advanced_insights = self._extract_advanced_insights(advanced_llm_response)
        spiritual_context = spiritual_enhancement.get("enhanced_context", "")
        
        # Combine insights with spiritual wisdom
        improved_response = f"""
{spiritual_context}

{dharma_insights}

{advanced_insights}

May this guidance serve your spiritual journey with wisdom and compassion. 🙏
        """.strip()
        
        return improved_response

    def _extract_spiritual_insights(self, response: str) -> str:
        """Extract spiritual insights from DharmaLLM response"""
        # Simple extraction - could be enhanced with NLP
        lines = response.split('\n')
        spiritual_lines = [line for line in lines if any(
            word in line.lower() for word in 
            ["meditation", "wisdom", "compassion", "dharma", "spiritual", "consciousness"]
        )]
        return '\n'.join(spiritual_lines[:3])  # Top 3 spiritual insights

    def _extract_advanced_insights(self, response: str) -> str:
        """Extract advanced insights from LLM response"""
        # Simple extraction - could be enhanced with NLP
        lines = response.split('\n')
        return '\n'.join(lines[:2])  # First 2 lines usually contain key insights

    async def _maybe_trigger_training(self):
        """Check if we should trigger a training cycle"""
        
        current_time = datetime.now()
        last_training_file = self.logs_dir / "last_training.json"
        
        should_train = False
        
        if not last_training_file.exists():
            should_train = True
        else:
            with open(last_training_file, 'r') as f:
                last_training = json.load(f)
                last_time = datetime.fromisoformat(last_training["timestamp"])
                hours_since = (current_time - last_time).total_seconds() / 3600
                
                if hours_since >= self.learning_config["retrain_interval_hours"]:
                    should_train = True
        
        if should_train:
            logging.info("🔄 Triggering continuous learning training...")
            await self.train_from_interactions()

    async def train_from_interactions(self):
        """Train the model from collected interactions"""
        
        if len(self.training_data) < self.learning_config["min_interactions_for_training"]:
            logging.info(f"Not enough training data: {len(self.training_data)} < {self.learning_config['min_interactions_for_training']}")
            return
        
        logging.info(f"🚀 Starting continuous learning with {len(self.training_data)} examples")
        
        try:
            # Prepare training dataset
            dataset = self._prepare_training_dataset()
            
            # Load base model
            model_path = self.model_manager.get_latest_model_path()
            if not model_path:
                logging.error("No base model found for training")
                return
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.models_dir / "continuous_learning"),
                num_train_epochs=2,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                warmup_steps=100,
                learning_rate=5e-5,
                logging_dir=str(self.logs_dir),
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="no",
                save_total_limit=3,
                load_best_model_at_end=False,
                metric_for_best_model="loss",
                greater_is_better=False,
                report_to="wandb"
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )
            
            # Train
            trainer.train()
            
            # Save the improved model
            new_model_path = self.models_dir / f"dharmallm_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trainer.save_model(str(new_model_path))
            
            # Update model manager
            self.model_manager.load_model(str(new_model_path))
            
            # Log training completion
            training_log = {
                "timestamp": datetime.now().isoformat(),
                "training_examples": len(self.training_data),
                "model_path": str(new_model_path),
                "training_args": training_args.to_dict()
            }
            
            with open(self.logs_dir / "last_training.json", 'w') as f:
                json.dump(training_log, f, indent=2)
            
            # Log to W&B
            wandb.log({
                "training_examples": len(self.training_data),
                "training_completed": 1,
                "model_version": new_model_path.name
            })
            
            logging.info(f"✅ Continuous learning completed. New model saved: {new_model_path}")
            
            # Clear training data to start fresh
            self.training_data = []
            
        except Exception as e:
            logging.error(f"❌ Continuous learning failed: {e}")
            wandb.log({"training_error": str(e)})

    def _prepare_training_dataset(self) -> Dataset:
        """Prepare training dataset from collected interactions"""
        
        # Sort by quality score and take the best examples
        sorted_data = sorted(self.training_data, 
                           key=lambda x: x["quality_score"], 
                           reverse=True)
        
        # Limit to max training examples
        max_examples = self.learning_config["max_training_examples"]
        if len(sorted_data) > max_examples:
            sorted_data = sorted_data[:max_examples]
        
        # Prepare for transformers
        inputs = []
        outputs = []
        
        for example in sorted_data:
            # Format as instruction-following
            input_text = f"User: {example['input']}\nDharmaLLM:"
            output_text = example['output']
            
            inputs.append(input_text)
            outputs.append(output_text)
        
        return Dataset.from_dict({
            "input": inputs,
            "output": outputs
        })

    async def _save_interaction_data(self):
        """Save interaction data to file"""
        
        # Save recent interactions
        interactions_file = self.data_dir / "interaction_history.json"
        
        # Keep only last 1000 interactions to manage file size
        recent_interactions = self.interaction_history[-1000:]
        
        with open(interactions_file, 'w') as f:
            json.dump(recent_interactions, f, indent=2)
        
        # Save training data
        training_file = self.data_dir / "training_data.json"
        with open(training_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)

    async def load_interaction_history(self):
        """Load previous interaction history"""
        
        interactions_file = self.data_dir / "interaction_history.json"
        training_file = self.data_dir / "training_data.json"
        
        if interactions_file.exists():
            with open(interactions_file, 'r') as f:
                self.interaction_history = json.load(f)
                logging.info(f"📚 Loaded {len(self.interaction_history)} previous interactions")
        
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)
                logging.info(f"🎯 Loaded {len(self.training_data)} training examples")

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning process"""
        
        return {
            "total_interactions": len(self.interaction_history),
            "training_examples": len(self.training_data),
            "average_quality_score": sum(ex["quality_score"] for ex in self.training_data) / len(self.training_data) if self.training_data else 0,
            "last_training": self._get_last_training_time(),
            "next_training_due": self._get_next_training_time(),
            "learning_config": self.learning_config
        }

    def _get_last_training_time(self) -> Optional[str]:
        """Get timestamp of last training"""
        
        last_training_file = self.logs_dir / "last_training.json"
        if last_training_file.exists():
            with open(last_training_file, 'r') as f:
                return json.load(f)["timestamp"]
        return None

    def _get_next_training_time(self) -> Optional[str]:
        """Calculate when next training is due"""
        
        last_training = self._get_last_training_time()
        if last_training:
            last_time = datetime.fromisoformat(last_training)
            next_time = last_time.replace(hour=last_time.hour + self.learning_config["retrain_interval_hours"])
            return next_time.isoformat()
        return None

    async def force_training(self):
        """Force immediate training regardless of schedule"""
        logging.info("🔧 Forcing immediate training...")
        await self.train_from_interactions()

    def __del__(self):
        """Cleanup when learner is destroyed"""
        if hasattr(self, 'wandb') and wandb.run:
            wandb.finish()
