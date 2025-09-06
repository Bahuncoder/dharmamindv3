"""
🕉️ DharmaLLM Enhanced Package - Complete AI Training & Deployment System

Enterprise-grade AI model training and deployment with deep dharmic intelligence.

Components:
- config: Advanced configuration management
- training: Enterprise training engine
- evaluate: Comprehensive evaluation framework  
- data: Advanced data processing pipeline
- models: Model management and deployment

May this system create truly wise and compassionate AI models.
"""

__version__ = "0.1.0"
__author__ = "DharmaMind Team"
__email__ = "team@dharmamind.ai"
__description__ = "AI with Soul powered by Dharma"

# Import core components
from .config.advanced_config import (
    DharmaLLMAdvancedConfig,
    DharmaLLMConfigFactory,
    ModelArchitecture,
    TrainingStage,
    WisdomTradition,
    DharmicPrinciple,
    EvaluationMetric,
    OptimizationObjective
)

from .training.enterprise_trainer import (
    DharmaLLMTrainingEngine,
    DharmicLossFunction,
    WisdomValidator,
    CulturalAdapter
)

from .evaluate.advanced_evaluator import (
    DharmaLLMAdvancedEvaluator,
    EvaluationResult,
    DharmicScore,
    WisdomAssessment,
    CulturalSensitivityScore,
    CompassionMetrics,
    SafetyAssessment
)

from .data.advanced_preprocessor import (
    DharmaLLMDataProcessor,
    DataQualityMetrics,
    ProcessedDataSample,
    TextCleaner,
    DharmicContentAnalyzer,
    CulturalSensitivityAnalyzer
)

from .models.model_manager import (
    ModelRegistry,
    ModelMetadata,
    DharmaLLMServingEngine,
    ModelDeploymentManager,
    QualityGateSystem,
    GenerationRequest,
    GenerationResponse
)

# Version and metadata
__version__ = "2.0.0"
__title__ = "DharmaLLM Enhanced"
__description__ = "Enterprise AI training with dharmic intelligence"
__author__ = "DharmaMind Team"

# Package-level constants
SUPPORTED_ARCHITECTURES = [
    ModelArchitecture.TRANSFORMER,
    ModelArchitecture.LLAMA,
    ModelArchitecture.MISTRAL,
    ModelArchitecture.DHARMA_GPT
]

DHARMIC_PRINCIPLES = [
    DharmicPrinciple.AHIMSA,
    DharmicPrinciple.SATYA,
    DharmicPrinciple.ASTEYA,
    DharmicPrinciple.BRAHMACHARYA,
    DharmicPrinciple.APARIGRAHA
]

WISDOM_TRADITIONS = [
    WisdomTradition.VEDANTIC,
    WisdomTradition.HINDU,
    WisdomTradition.TANTRIC,
    WisdomTradition.AYURVEDIC
]

# Convenience functions
def create_config(config_type="development", **kwargs):
    """Create DharmaLLM configuration"""
    return DharmaLLMConfigFactory.create_config(config_type, **kwargs)

def create_trainer(config):
    """Create training engine"""
    return DharmaLLMTrainingEngine(config)

def create_evaluator(config):
    """Create evaluation engine"""
    return DharmaLLMAdvancedEvaluator(config)

def create_data_processor(config):
    """Create data processing pipeline"""
    return DharmaLLMDataProcessor(config)

def create_model_manager(config):
    """Create model management system"""
    registry = ModelRegistry("models/registry")
    return ModelDeploymentManager(config, registry)

# Quick start function
def quick_start(config_type="development"):
    """Quick start DharmaLLM system"""
    config = create_config(config_type)
    
    return {
        "config": config,
        "trainer": create_trainer(config),
        "evaluator": create_evaluator(config),
        "data_processor": create_data_processor(config),
        "model_manager": create_model_manager(config)
    }

__all__ = [
    # Configuration
    "DharmaLLMAdvancedConfig",
    "DharmaLLMConfigFactory", 
    "ModelArchitecture",
    "TrainingStage",
    "WisdomTradition",
    "DharmicPrinciple",
    "EvaluationMetric",
    "OptimizationObjective",
    
    # Training
    "DharmaLLMTrainingEngine",
    "DharmicLossFunction",
    "WisdomValidator",
    "CulturalAdapter",
    
    # Evaluation
    "DharmaLLMAdvancedEvaluator",
    "EvaluationResult",
    "DharmicScore",
    "WisdomAssessment", 
    "CulturalSensitivityScore",
    "CompassionMetrics",
    "SafetyAssessment",
    
    # Data Processing
    "DharmaLLMDataProcessor",
    "DataQualityMetrics",
    "ProcessedDataSample",
    "TextCleaner",
    "DharmicContentAnalyzer",
    "CulturalSensitivityAnalyzer",
    
    # Model Management
    "ModelRegistry",
    "ModelMetadata",
    "DharmaLLMServingEngine",
    "ModelDeploymentManager",
    "QualityGateSystem",
    "GenerationRequest",
    "GenerationResponse",
    
    # Convenience
    "create_config",
    "create_trainer",
    "create_evaluator", 
    "create_data_processor",
    "create_model_manager",
    "quick_start",
    
    # Constants
    "SUPPORTED_ARCHITECTURES",
    "DHARMIC_PRINCIPLES",
    "WISDOM_TRADITIONS"
]
