"""
Generate Embeddings Script

Generates vector embeddings for all content in the system.
Supports multiple embedding models and batch processing.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for text content"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        
    async def initialize(self):
        """Initialize embedding model"""
        try:
            # In real implementation, would load sentence-transformers model
            # from sentence_transformers import SentenceTransformer
            # self.model = SentenceTransformer(self.model_name)
            
            # For now, use placeholder
            self.model = "placeholder_model"
            logger.info(f"Embedding model initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        
        # Placeholder implementation (would use real model)
        # return self.model.encode(text).tolist()
        
        # Simple hash-based embedding for demo
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        embedding = []
        for i in range(self.embedding_dim):
            embedding.append(float((hash_int >> (i % 32)) & 1) * 2 - 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        
        # In real implementation, would use batch processing
        # return self.model.encode(texts).tolist()
        
        return [self.generate_embedding(text) for text in texts]

async def load_module_content(modules_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all module content for embedding generation"""
    
    modules_content = {}
    
    for yaml_file in modules_dir.glob("*.yaml"):
        try:
            import yaml
            with open(yaml_file, 'r', encoding='utf-8') as f:
                module_data = yaml.safe_load(f)
            
            module_name = module_data['name']
            
            # Combine all text content
            content_texts = [
                module_data['description'],
                ' '.join(module_data.get('expertise_areas', [])),
                ' '.join(module_data.get('guidance_patterns', [])),
                ' '.join(module_data.get('when_to_use', []))
            ]
            
            modules_content[module_name] = {
                'texts': content_texts,
                'metadata': {
                    'category': module_data['category'],
                    'expertise_areas': module_data.get('expertise_areas', []),
                    'file_path': str(yaml_file)
                }
            }
            
            logger.debug(f"Loaded content for module: {module_name}")
            
        except Exception as e:
            logger.error(f"Error loading {yaml_file}: {e}")
    
    logger.info(f"Loaded content for {len(modules_content)} modules")
    return modules_content

async def load_conversation_data(data_dir: Path) -> List[Dict[str, Any]]:
    """Load conversation data for embedding generation"""
    
    conversations = []
    
    # Look for conversation data files
    for json_file in data_dir.glob("conversations*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                conversations.extend(data)
            else:
                conversations.append(data)
                
            logger.debug(f"Loaded {len(data)} conversations from {json_file}")
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations

async def generate_module_embeddings(
    generator: EmbeddingGenerator,
    modules_content: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Generate embeddings for all module content"""
    
    module_embeddings = {}
    
    for module_name, content in modules_content.items():
        try:
            logger.info(f"Generating embeddings for module: {module_name}")
            
            # Generate embeddings for each text segment
            embeddings = generator.generate_embeddings_batch(content['texts'])
            
            # Calculate average embedding for the module
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0).tolist()
            else:
                avg_embedding = [0.0] * generator.embedding_dim
            
            module_embeddings[module_name] = {
                'embedding': avg_embedding,
                'segment_embeddings': embeddings,
                'metadata': content['metadata'],
                'embedding_model': generator.model_name,
                'dimension': generator.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {module_name}: {e}")
    
    return module_embeddings

async def generate_conversation_embeddings(
    generator: EmbeddingGenerator,
    conversations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate embeddings for conversation data"""
    
    conversation_embeddings = []
    
    for i, conv in enumerate(conversations):
        try:
            if i % 100 == 0:
                logger.info(f"Processing conversation {i+1}/{len(conversations)}")
            
            # Extract text content
            user_message = conv.get('user_message', '')
            ai_response = conv.get('ai_response', '')
            
            if not user_message and not ai_response:
                continue
            
            # Generate embeddings
            user_embedding = generator.generate_embedding(user_message) if user_message else None
            response_embedding = generator.generate_embedding(ai_response) if ai_response else None
            
            conversation_embeddings.append({
                'conversation_id': conv.get('conversation_id', f'conv_{i}'),
                'user_message': user_message,
                'ai_response': ai_response,
                'user_embedding': user_embedding,
                'response_embedding': response_embedding,
                'metadata': {
                    'timestamp': conv.get('timestamp'),
                    'modules_used': conv.get('modules_used', []),
                    'confidence_score': conv.get('confidence_score'),
                    'dharmic_alignment': conv.get('dharmic_alignment')
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing conversation {i}: {e}")
    
    logger.info(f"Generated embeddings for {len(conversation_embeddings)} conversations")
    return conversation_embeddings

async def save_embeddings(
    module_embeddings: Dict[str, Dict[str, Any]],
    conversation_embeddings: List[Dict[str, Any]],
    output_dir: Path
):
    """Save embeddings to files"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save module embeddings
    modules_file = output_dir / "module_embeddings.json"
    with open(modules_file, 'w', encoding='utf-8') as f:
        json.dump(module_embeddings, f, indent=2)
    
    logger.info(f"Saved module embeddings: {modules_file}")
    
    # Save conversation embeddings
    conversations_file = output_dir / "conversation_embeddings.json"
    with open(conversations_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_embeddings, f, indent=2)
    
    logger.info(f"Saved conversation embeddings: {conversations_file}")
    
    # Save metadata
    metadata = {
        'embedding_model': module_embeddings.get(list(module_embeddings.keys())[0], {}).get('embedding_model', 'unknown') if module_embeddings else 'unknown',
        'dimension': module_embeddings.get(list(module_embeddings.keys())[0], {}).get('dimension', 384) if module_embeddings else 384,
        'module_count': len(module_embeddings),
        'conversation_count': len(conversation_embeddings),
        'generated_at': str(asyncio.get_event_loop().time())
    }
    
    metadata_file = output_dir / "embeddings_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata: {metadata_file}")

async def create_vector_index(embeddings_dir: Path, index_type: str = "faiss"):
    """Create vector search index"""
    
    try:
        # Load module embeddings
        modules_file = embeddings_dir / "module_embeddings.json"
        if modules_file.exists():
            with open(modules_file, 'r') as f:
                module_embeddings = json.load(f)
            
            # Extract vectors and create index
            vectors = []
            labels = []
            
            for module_name, data in module_embeddings.items():
                vectors.append(data['embedding'])
                labels.append(module_name)
            
            if vectors:
                # In real implementation, would create FAISS or similar index
                index_data = {
                    'vectors': vectors,
                    'labels': labels,
                    'dimension': len(vectors[0]),
                    'index_type': index_type
                }
                
                index_file = embeddings_dir / f"vector_index_{index_type}.json"
                with open(index_file, 'w') as f:
                    json.dump(index_data, f, indent=2)
                
                logger.info(f"Created vector index: {index_file}")
        
    except Exception as e:
        logger.error(f"Error creating vector index: {e}")

async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Generate embeddings for DharmaMind content")
    parser.add_argument("--modules-dir", type=Path, default="./modules",
                       help="Directory containing module YAML files")
    parser.add_argument("--data-dir", type=Path, default="./data",
                       help="Directory containing conversation data")
    parser.add_argument("--output-dir", type=Path, default="./embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model to use")
    parser.add_argument("--skip-conversations", action="store_true",
                       help="Skip conversation embeddings")
    parser.add_argument("--create-index", action="store_true",
                       help="Create vector search index")
    
    args = parser.parse_args()
    
    try:
        # Initialize embedding generator
        generator = EmbeddingGenerator(args.model)
        await generator.initialize()
        
        # Load module content
        logger.info("Loading module content...")
        modules_content = await load_module_content(args.modules_dir)
        
        # Generate module embeddings
        logger.info("Generating module embeddings...")
        module_embeddings = await generate_module_embeddings(generator, modules_content)
        
        # Load and process conversations if requested
        conversation_embeddings = []
        if not args.skip_conversations:
            logger.info("Loading conversation data...")
            conversations = await load_conversation_data(args.data_dir)
            
            if conversations:
                logger.info("Generating conversation embeddings...")
                conversation_embeddings = await generate_conversation_embeddings(generator, conversations)
        
        # Save all embeddings
        logger.info("Saving embeddings...")
        await save_embeddings(module_embeddings, conversation_embeddings, args.output_dir)
        
        # Create vector index if requested
        if args.create_index:
            logger.info("Creating vector index...")
            await create_vector_index(args.output_dir)
        
        logger.info("Embedding generation completed successfully")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Module embeddings: {len(module_embeddings)}")
        print(f"  Conversation embeddings: {len(conversation_embeddings)}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Embedding model: {args.model}")
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
