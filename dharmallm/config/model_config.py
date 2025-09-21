"""
🕉️ DharmaLLM Advanced Configuration - Complete System

Enterprise-grade configuration management for dharmic AI model training and deployment:

Core Configuration Areas:
- Model Architecture and Training Parameters
- Dharmic Alignment and Wisdom Scoring
- Multi-modal Learning and Fine-tuning
- Distributed Training and Scaling
- Evaluation Metrics and Benchmarking
- Data Pipeline and Preprocessing
- Model Serving and Deployment
- Monitoring and Observability

Advanced Features:
- Multi-stage training pipelines
- Dharmic principle integration
- Cultural sensitivity training
- Wisdom tradition specialization
- Real-time model adaptation
- Continuous learning systems
- Performance optimization
- Ethical AI compliance

May this configuration guide the creation of truly wise AI 🔧
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION ENUMS
# ===============================

class ModelArchitecture(Enum):
    TRANSFORMER = "transformer"
    LLAMA = "llama"
    MISTRAL = "mistral"
    PALM = "palm"
    DHARMA_GPT = "dharma_gpt"
    WISDOM_BERT = "wisdom_bert"

class TrainingStage(Enum):
    PRETRAINING = "pretraining"
    FINE_TUNING = "fine_tuning"
    DHARMIC_ALIGNMENT = "dharmic_alignment"
    WISDOM_SPECIALIZATION = "wisdom_specialization"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class WisdomTradition(Enum):
    VEDANTIC = "vedantic"       # Advaita, Dvaita, Vishishtadvaita
    SAMKHYA = "samkhya"         # Kapila's philosophy
    YOGA = "yoga"               # Patanjali's system
    TANTRIC = "tantric"         # Shaiva and Shakta traditions
    AYURVEDIC = "ayurvedic"     # Traditional medicine
    HINDU = "hindu"             # General Hindu wisdom
    PURANIC = "puranic"         # Purana-based knowledge
    VEDIC = "vedic"             # Vedic hymns and rituals

class DharmicPrinciple(Enum):
    AHIMSA = "ahimsa"           # Non-violence
    SATYA = "satya"             # Truthfulness
    ASTEYA = "asteya"           # Non-stealing
    BRAHMACHARYA = "brahmacharya"  # Moderation
    APARIGRAHA = "aparigraha"   # Non-possessiveness

class EvaluationMetric(Enum):
    DHARMIC_ALIGNMENT = "dharmic_alignment"
    WISDOM_DEPTH = "wisdom_depth"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    COMPASSION_LEVEL = "compassion_level"
    TRUTHFULNESS = "truthfulness"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    PERPLEXITY = "perplexity"
    BLEU_SCORE = "bleu_score"

class OptimizationObjective(Enum):
    MAXIMIZE_WISDOM = "maximize_wisdom"
    ENHANCE_COMPASSION = "enhance_compassion"
    IMPROVE_TRUTHFULNESS = "improve_truthfulness"
    REDUCE_HARM = "reduce_harm"
    CULTURAL_ALIGNMENT = "cultural_alignment"
    BALANCED_DHARMA = "balanced_dharma"

# ===============================
# ADVANCED CONFIGURATION CLASSES
# ===============================

@dataclass
class ModelConfig:
    """Configuration for model architecture and parameters"""
    
    # Architecture
    architecture: ModelArchitecture = ModelArchitecture.TRANSFORMER
    model_size: str = "7b"  # 1b, 7b, 13b, 30b, 65b
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    vocab_size: int = 32000
    
    # Dharmic enhancements
    dharmic_embedding_size: int = 256
    wisdom_head_size: int = 128
    cultural_context_size: int = 64
    principle_alignment_layers: int = 4
    
    # Specialized components
    enable_wisdom_router: bool = True
    enable_cultural_adapter: bool = True
    enable_principle_checker: bool = True
    enable_compassion_amplifier: bool = True
    
    # Model variants
    base_model_path: Optional[str] = None
    dharmic_pretrained_path: Optional[str] = None
    wisdom_embeddings_path: Optional[str] = None

@dataclass
class TrainingConfig:
    """Configuration for training parameters and strategies"""
    
    # Basic training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    
    # Advanced optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clipping: float = 1.0
    fp16: bool = True
    bf16: bool = False
    
    # Dharmic training enhancements
    dharmic_loss_weight: float = 0.3
    wisdom_consistency_weight: float = 0.2
    cultural_sensitivity_weight: float = 0.15
    principle_alignment_weight: float = 0.25
    compassion_reward_weight: float = 0.1
    
    # Multi-stage training
    training_stages: List[TrainingStage] = field(default_factory=lambda: [
        TrainingStage.FINE_TUNING,
        TrainingStage.DHARMIC_ALIGNMENT,
        TrainingStage.WISDOM_SPECIALIZATION
    ])
    
    # Curriculum learning
    enable_curriculum_learning: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Reinforcement learning
    enable_rlhf: bool = True
    reward_model_path: Optional[str] = None
    ppo_epochs: int = 4
    kl_penalty: float = 0.1

@dataclass
class DataConfig:
    """Configuration for data sources and processing"""
    
    # Data sources
    primary_data_path: str = "data/processed"
    wisdom_corpus_path: str = "data/wisdom_corpus"
    scripture_data_path: str = "data/scriptures"
    conversation_data_path: str = "data/conversations"
    cultural_data_path: str = "data/cultural"
    
    # Data categories
    wisdom_traditions: List[WisdomTradition] = field(default_factory=lambda: [
        WisdomTradition.VEDANTIC,
        WisdomTradition.HINDU,
        WisdomTradition.VEDIC
    ])
    
    # Data quality thresholds
    min_dharmic_score: float = 0.6
    min_wisdom_score: float = 0.5
    min_cultural_sensitivity: float = 0.7
    max_toxicity_score: float = 0.1
    
    # Data augmentation
    enable_data_augmentation: bool = True
    augmentation_strategies: List[str] = field(default_factory=lambda: [
        "paraphrasing",
        "cultural_translation",
        "wisdom_expansion",
        "principle_alignment"
    ])
    
    # Preprocessing
    max_sequence_length: int = 2048
    truncation_strategy: str = "longest_first"
    padding_strategy: str = "max_length"
    remove_duplicates: bool = True
    normalize_text: bool = True

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and metrics"""
    
    # Evaluation datasets
    eval_datasets: List[str] = field(default_factory=lambda: [
        "dharmic_qa",
        "wisdom_generation",
        "cultural_sensitivity",
        "principle_alignment",
        "compassion_test"
    ])
    
    # Evaluation metrics
    primary_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.DHARMIC_ALIGNMENT,
        EvaluationMetric.WISDOM_DEPTH,
        EvaluationMetric.CULTURAL_SENSITIVITY
    ])
    
    # Benchmarking
    enable_human_evaluation: bool = True
    human_eval_sample_size: int = 1000
    expert_evaluator_count: int = 5
    
    # Automated evaluation
    eval_frequency: str = "epoch"  # batch, epoch, steps
    eval_steps: int = 500
    save_best_model: bool = True
    early_stopping_patience: int = 3
    
    # Dharmic evaluation specifics
    dharmic_principle_weights: Dict[str, float] = field(default_factory=lambda: {
        "ahimsa": 0.25,
        "satya": 0.25,
        "asteya": 0.15,
        "brahmacharya": 0.15,
        "aparigraha": 0.20
    })

@dataclass
class DeploymentConfig:
    """Configuration for model deployment and serving"""
    
    # Serving configuration
    model_server: str = "vllm"  # vllm, text-generation-inference, transformers
    max_concurrent_requests: int = 100
    max_sequence_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Scalability
    enable_model_parallelism: bool = True
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_quantization: bool = True
    quantization_method: str = "int8"  # int8, int4, gptq, awq
    
    # Monitoring
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_collection: bool = True
    prometheus_metrics: bool = True
    
    # Safety and compliance
    enable_content_filtering: bool = True
    max_response_length: int = 2048
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    
    # Dharmic safeguards
    enable_dharmic_filtering: bool = True
    min_dharmic_threshold: float = 0.6
    enable_cultural_check: bool = True
    enable_principle_validation: bool = True

@dataclass
class ExperimentConfig:
    """Configuration for experiments and hyperparameter tuning"""
    
    # Experiment tracking
    experiment_name: str = "dharmallm_experiment"
    run_name: Optional[str] = None
    project_name: str = "dharmallm"
    
    # Tracking tools
    use_wandb: bool = True
    use_tensorboard: bool = True
    use_mlflow: bool = False
    
    # Hyperparameter search
    enable_hyperparameter_search: bool = False
    search_strategy: str = "random"  # random, grid, bayesian
    num_trials: int = 20
    
    # Search spaces
    learning_rate_range: Tuple[float, float] = (1e-6, 1e-3)
    batch_size_options: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    dharmic_weight_range: Tuple[float, float] = (0.1, 0.5)
    
    # Reproducibility
    random_seed: int = 42
    deterministic_training: bool = True

@dataclass
class DharmaLLMAdvancedConfig:
    """Complete configuration for DharmaLLM system"""
    
    # Core configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # System configuration
    output_dir: str = "models/dharmallm"
    cache_dir: str = "cache"
    log_dir: str = "logs"
    data_dir: str = "data"
    
    # Hardware configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    num_gpus: int = 1
    gpu_memory_limit: Optional[int] = None
    enable_gradient_checkpointing: bool = True
    
    # Dharmic system metadata
    dharmic_version: str = "2.0.0"
    wisdom_model_version: str = "1.0.0"
    cultural_adaptation_version: str = "1.0.0"
    
    # Optimization objectives
    primary_objective: OptimizationObjective = OptimizationObjective.BALANCED_DHARMA
    secondary_objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.ENHANCE_COMPASSION,
        OptimizationObjective.IMPROVE_TRUTHFULNESS
    ])
    
    def save_config(self, path: str):
        """Save configuration to file"""
        config_dict = asdict(self)
        
        # Convert enums to strings for serialization
        config_dict = self._convert_enums_to_strings(config_dict)
        
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {path}")
    
    @classmethod
    def load_config(cls, path: str) -> 'DharmaLLMAdvancedConfig':
        """Load configuration from file"""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")
        
        # Convert string values back to enums
        config_dict = cls._convert_strings_to_enums(config_dict)
        
        return cls(**config_dict)
    
    def _convert_enums_to_strings(self, obj):
        """Recursively convert enum values to strings"""
        if isinstance(obj, dict):
            return {k: self._convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_to_strings(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj
    
    @classmethod
    def _convert_strings_to_enums(cls, obj):
        """Recursively convert string values back to enums"""
        # This would require more sophisticated logic to map strings back to enums
        # For now, return as-is
        return obj
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate model configuration
        if self.model.hidden_size <= 0:
            issues.append("Model hidden_size must be positive")
        
        if self.model.num_layers <= 0:
            issues.append("Model num_layers must be positive")
        
        # Validate training configuration
        if self.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        # Validate dharmic weights sum to reasonable value
        dharmic_weights_sum = (
            self.training.dharmic_loss_weight +
            self.training.wisdom_consistency_weight +
            self.training.cultural_sensitivity_weight +
            self.training.principle_alignment_weight +
            self.training.compassion_reward_weight
        )
        
        if dharmic_weights_sum > 1.5:
            issues.append("Sum of dharmic loss weights is too high")
        
        # Validate data configuration
        if not os.path.exists(self.data.primary_data_path):
            issues.append(f"Primary data path does not exist: {self.data.primary_data_path}")
        
        # Validate evaluation configuration
        if self.evaluation.early_stopping_patience <= 0:
            issues.append("Early stopping patience must be positive")
        
        return issues
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_dir,
            self.cache_dir,
            self.log_dir,
            self.data_dir,
            f"{self.output_dir}/checkpoints",
            f"{self.output_dir}/final_model",
            f"{self.log_dir}/training",
            f"{self.log_dir}/evaluation",
            f"{self.data_dir}/processed",
            f"{self.data_dir}/raw"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

# ===============================
# CONFIGURATION PRESETS
# ===============================

def get_development_config() -> DharmaLLMAdvancedConfig:
    """Get configuration for development environment"""
    config = DharmaLLMAdvancedConfig()
    
    # Smaller model for development
    config.model.model_size = "1b"
    config.model.hidden_size = 2048
    config.model.num_layers = 16
    
    # Faster training for development
    config.training.batch_size = 4
    config.training.num_epochs = 1
    config.training.max_steps = 1000
    
    # Limited evaluation for speed
    config.evaluation.eval_steps = 100
    config.evaluation.human_eval_sample_size = 100
    
    return config

def get_production_config() -> DharmaLLMAdvancedConfig:
    """Get configuration for production training"""
    config = DharmaLLMAdvancedConfig()
    
    # Full-size model for production
    config.model.model_size = "7b"
    config.model.hidden_size = 4096
    config.model.num_layers = 32
    
    # Comprehensive training
    config.training.num_epochs = 5
    config.training.enable_rlhf = True
    config.training.enable_curriculum_learning = True
    
    # Thorough evaluation
    config.evaluation.enable_human_evaluation = True
    config.evaluation.human_eval_sample_size = 5000
    config.evaluation.expert_evaluator_count = 10
    
    return config

def get_research_config() -> DharmaLLMAdvancedConfig:
    """Get configuration for research experiments"""
    config = DharmaLLMAdvancedConfig()
    
    # Enable experimental features
    config.experiment.enable_hyperparameter_search = True
    config.experiment.num_trials = 50
    
    # Multiple training stages
    config.training.training_stages = [
        TrainingStage.PRETRAINING,
        TrainingStage.FINE_TUNING,
        TrainingStage.DHARMIC_ALIGNMENT,
        TrainingStage.WISDOM_SPECIALIZATION,
        TrainingStage.CULTURAL_ADAPTATION,
        TrainingStage.REINFORCEMENT_LEARNING
    ]
    
    # Comprehensive evaluation
    config.evaluation.primary_metrics = [
        EvaluationMetric.DHARMIC_ALIGNMENT,
        EvaluationMetric.WISDOM_DEPTH,
        EvaluationMetric.CULTURAL_SENSITIVITY,
        EvaluationMetric.COMPASSION_LEVEL,
        EvaluationMetric.TRUTHFULNESS,
        EvaluationMetric.SAFETY
    ]
    
    return config

# ===============================
# CONFIGURATION FACTORY
# ===============================

class DharmaLLMConfigFactory:
    """Factory for creating DharmaLLM configurations"""
    
    @staticmethod
    def create_config(
        config_type: str = "development",
        **overrides
    ) -> DharmaLLMAdvancedConfig:
        """Create configuration with optional overrides"""
        
        if config_type == "development":
            config = get_development_config()
        elif config_type == "production":
            config = get_production_config()
        elif config_type == "research":
            config = get_research_config()
        else:
            config = DharmaLLMAdvancedConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Validate configuration
        issues = config.validate_config()
        if issues:
            logger.warning(f"Configuration issues found: {issues}")
        
        return config
    
    @staticmethod
    def create_wisdom_specialized_config(
        tradition: WisdomTradition
    ) -> DharmaLLMAdvancedConfig:
        """Create configuration specialized for specific wisdom tradition"""
        config = DharmaLLMAdvancedConfig()
        
        # Specialize for tradition
        config.data.wisdom_traditions = [tradition]
        config.training.training_stages.append(TrainingStage.WISDOM_SPECIALIZATION)
        
        # Adjust data paths
        config.data.primary_data_path = f"data/{tradition.value}"
        config.data.wisdom_corpus_path = f"data/wisdom/{tradition.value}"
        
        # Tradition-specific weights
        if tradition == WisdomTradition.VEDANTIC:
            config.training.wisdom_consistency_weight = 0.4
        elif tradition == WisdomTradition.TANTRIC:
            config.training.dharmic_loss_weight = 0.4
        elif tradition == WisdomTradition.YOGA:
            config.training.principle_alignment_weight = 0.4
        
        return config

# ===============================
# EXAMPLE USAGE
# ===============================

class AdvancedConfig:
    """Simple configuration class for continuous learning compatibility"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        
        # Model configuration
        self.model_config = {
            "model_name": "dharmallm-base",
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        }
        
        # Training configuration
        self.training_config = {
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "warmup_steps": 100,
            "save_steps": 500,
            "eval_steps": 250,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01
        }
        
        # Continuous learning configuration
        self.learning_config = {
            "min_interactions": 50,
            "quality_threshold": 4.0,
            "retrain_interval_hours": 24,
            "max_training_examples": 10000
        }
        
        # Paths
        self.paths = {
            "data_dir": self.base_path / "data",
            "models_dir": self.base_path / "models", 
            "logs_dir": self.base_path / "logs",
            "cache_dir": self.base_path / ".cache"
        }
        
        # Create directories
        for path in self.paths.values():
            path.mkdir(exist_ok=True)
            
        # W&B configuration
        self.wandb_config = {
            "project": "dharmallm",
            "entity": os.getenv("WANDB_ENTITY"),
            "api_key": os.getenv("WANDB_API_KEY"),
            "mode": "online" if os.getenv("WANDB_API_KEY") else "disabled"
        }
        
    def get_model_config(self):
        """Get model configuration"""
        return self.model_config
        
    def get_training_config(self):
        """Get training configuration"""
        return self.training_config
        
    def get_learning_config(self):
        """Get continuous learning configuration"""
        return self.learning_config
        
    def get_paths(self):
        """Get system paths"""
        return self.paths
        
    def get_wandb_config(self):
        """Get W&B configuration"""
        return self.wandb_config

if __name__ == "__main__":
    # Create development configuration
    dev_config = DharmaLLMConfigFactory.create_config("development")
    dev_config.create_directories()
    dev_config.save_config("configs/development.yaml")
    
    # Create production configuration
    prod_config = DharmaLLMConfigFactory.create_config("production")
    prod_config.save_config("configs/production.yaml")
    
    # Create wisdom-specialized configuration
    buddhist_config = DharmaLLMConfigFactory.create_wisdom_specialized_config(
        WisdomTradition.BUDDHIST
    )
    buddhist_config.save_config("configs/buddhist_specialized.yaml")
    
    logger.info("Configuration files created successfully!")
