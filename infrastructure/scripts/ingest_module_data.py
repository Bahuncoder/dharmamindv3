"""
Ingest Module Data Script

Loads Dharma module data from various sources into the system.
Processes YAML module definitions and validates them.
"""

import asyncio
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def load_module_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load module data from YAML file"""
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            module_data = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['name', 'description', 'category', 'expertise_areas']
        for field in required_fields:
            if field not in module_data:
                raise ValueError(f"Missing required field: {field}")
        
        logger.info(f"Loaded module: {module_data['name']}")
        return module_data
        
    except Exception as e:
        logger.error(f"Error loading {yaml_path}: {e}")
        return None

async def create_sample_modules(output_dir: Path):
    """Create sample module YAML files"""
    
    sample_modules = [
        {
            "name": "karma",
            "description": "Understanding of karma, action, and consequence in dharmic traditions",
            "category": "dharmic_principles",
            "expertise_areas": [
                "action_consequence",
                "moral_responsibility", 
                "past_actions",
                "future_outcomes",
                "ethical_decision_making",
                "cause_and_effect"
            ],
            "scriptural_references": [
                "Bhagavad Gita 2.47",
                "Bhagavad Gita 4.17",
                "Yoga Sutras 2.12"
            ],
            "guidance_patterns": [
                "Every action has consequences",
                "Consider the long-term effects of your choices",
                "Act with awareness and intention",
                "Take responsibility for your actions"
            ],
            "when_to_use": [
                "moral_dilemmas",
                "decision_making",
                "understanding_consequences",
                "ethical_questions",
                "life_choices"
            ]
        },
        {
            "name": "meditation",
            "description": "Mindfulness, meditation practices, and inner peace cultivation",
            "category": "spiritual_practice", 
            "expertise_areas": [
                "mindfulness",
                "concentration",
                "awareness",
                "inner_peace",
                "mental_training",
                "breath_awareness",
                "present_moment",
                "observation"
            ],
            "scriptural_references": [
                "Yoga Sutras 1.2",
                "Yoga Sutras 3.1-3",
                "Dhammapada 372"
            ],
            "guidance_patterns": [
                "Begin with breath awareness",
                "Observe thoughts without judgment",
                "Return attention to the present moment",
                "Practice regularly, even for short periods"
            ],
            "when_to_use": [
                "stress_anxiety",
                "spiritual_practice",
                "mental_clarity",
                "emotional_regulation",
                "inner_peace_seeking"
            ]
        },
        {
            "name": "compassion",
            "description": "Loving-kindness, empathy, and compassion for all beings",
            "category": "virtues",
            "expertise_areas": [
                "loving_kindness",
                "empathy",
                "care_for_others",
                "understanding",
                "forgiveness",
                "universal_love",
                "emotional_support",
                "healing"
            ],
            "scriptural_references": [
                "Metta Sutta",
                "Bhagavad Gita 12.13",
                "Yoga Sutras 1.33"
            ],
            "guidance_patterns": [
                "Extend kindness to yourself first",
                "See the common humanity in all beings",
                "Practice loving-kindness meditation",
                "Respond to suffering with compassion"
            ],
            "when_to_use": [
                "relationship_conflicts",
                "emotional_pain",
                "anger_resentment",
                "healing_trauma",
                "social_connection"
            ]
        },
        {
            "name": "wisdom",
            "description": "Ancient wisdom, spiritual insights, and life guidance",
            "category": "knowledge",
            "expertise_areas": [
                "spiritual_insight",
                "life_guidance",
                "understanding",
                "discernment",
                "truth_seeking",
                "ancient_knowledge",
                "practical_wisdom",
                "enlightenment"
            ],
            "scriptural_references": [
                "Bhagavad Gita 2.50",
                "Katha Upanishad 1.2.23",
                "Dhammapada 282"
            ],
            "guidance_patterns": [
                "Seek understanding beyond appearances",
                "Learn from experience and reflection",
                "Question assumptions with open mind",
                "Apply wisdom in daily life"
            ],
            "when_to_use": [
                "life_confusion",
                "seeking_understanding",
                "philosophical_questions",
                "spiritual_growth",
                "complex_decisions"
            ]
        }
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for module_data in sample_modules:
        yaml_path = output_dir / f"{module_data['name']}.yaml"
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(module_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created module file: {yaml_path}")

async def validate_modules(modules_dir: Path) -> List[Dict[str, Any]]:
    """Validate all modules in directory"""
    
    valid_modules = []
    
    for yaml_file in modules_dir.glob("*.yaml"):
        module_data = await load_module_from_yaml(yaml_file)
        if module_data:
            valid_modules.append(module_data)
    
    logger.info(f"Validated {len(valid_modules)} modules")
    return valid_modules

async def generate_module_index(modules: List[Dict[str, Any]], output_path: Path):
    """Generate module index for quick lookup"""
    
    index = {
        "modules": {},
        "categories": {},
        "expertise_areas": {},
        "total_modules": len(modules)
    }
    
    for module in modules:
        name = module['name']
        category = module['category']
        
        # Add to modules index
        index["modules"][name] = {
            "description": module['description'],
            "category": category,
            "expertise_count": len(module['expertise_areas'])
        }
        
        # Add to categories index
        if category not in index["categories"]:
            index["categories"][category] = []
        index["categories"][category].append(name)
        
        # Add to expertise areas index
        for area in module['expertise_areas']:
            if area not in index["expertise_areas"]:
                index["expertise_areas"][area] = []
            index["expertise_areas"][area].append(name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Generated module index: {output_path}")

async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Ingest Dharma module data")
    parser.add_argument("--modules-dir", type=Path, default="./modules", 
                       help="Directory containing module YAML files")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample module files")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing modules")
    parser.add_argument("--generate-index", action="store_true",
                       help="Generate module index file")
    
    args = parser.parse_args()
    
    modules_dir = args.modules_dir
    
    try:
        if args.create_samples:
            logger.info("Creating sample module files...")
            await create_sample_modules(modules_dir)
        
        if args.validate:
            logger.info("Validating modules...")
            valid_modules = await validate_modules(modules_dir)
            
            if args.generate_index:
                index_path = modules_dir / "module_index.json"
                await generate_module_index(valid_modules, index_path)
        
        logger.info("Module data ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during module ingestion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
