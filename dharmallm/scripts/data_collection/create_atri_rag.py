#!/usr/bin/env python3
"""
Atri RAG System - Retrieval-Augmented Generation for Meditation Master
========================================================================

Creates vector embeddings and RAG system for Maharishi Atri's meditation knowledge.

Features:
- Vector database using ChromaDB
- Semantic search across Yoga Sutras, Upanishads, techniques
- Context-aware retrieval
- Citation tracking
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed.")
    print("   Install with: pip install sentence-transformers")


class AtriKnowledgeRAG:
    """RAG system for Atri's meditation knowledge"""
    
    def __init__(self, knowledge_dir: str = "data/rishi_knowledge/atri"):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_path = Path("engines/rishi/rag_systems/atri_vector_db")
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🔄 Loading embedding model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        else:
            self.embedder = None
            
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path)
            )
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="atri_meditation_knowledge",
                metadata={"description": "Atri's meditation wisdom"}
            )
            print(f"✅ Vector database initialized at {self.db_path}")
        else:
            self.client = None
            self.collection = None
    
    def load_yoga_sutras(self) -> List[Dict[str, Any]]:
        """Load structured Yoga Sutras"""
        print("\n📖 Loading Yoga Sutras...")
        
        sutras_file = (self.knowledge_dir / "primary_texts" / 
                      "yoga_sutras_structured.json")
        
        if not sutras_file.exists():
            print(f"❌ Yoga Sutras not found at {sutras_file}")
            return []
        
        with open(sutras_file, 'r', encoding='utf-8') as f:
            yoga_sutras = json.load(f)
        
        documents = []
        for book_key, book_data in yoga_sutras.items():
            for sutra in book_data.get('sutras', []):
                # Create searchable document
                doc_text = f"""
Patanjali Yoga Sutra {sutra['number']}
{book_data['title']}

Sanskrit: {sutra['sanskrit']}
Transliteration: {sutra['transliteration']}
Translation: {sutra['translation']}

Commentary: {sutra['commentary']}
                """.strip()
                
                documents.append({
                    'text': doc_text,
                    'metadata': {
                        'source': 'Patanjali Yoga Sutras',
                        'sutra_number': sutra['number'],
                        'book': book_data['title'],
                        'sanskrit': sutra['sanskrit'],
                        'category': 'primary_text',
                        'type': 'sutra'
                    }
                })
        
        print(f"✅ Loaded {len(documents)} Yoga Sutras")
        return documents
    
    def load_upanishads(self) -> List[Dict[str, Any]]:
        """Load Upanishad texts"""
        print("\n📚 Loading Upanishads...")
        
        upanishads_dir = self.knowledge_dir / "upanishads"
        if not upanishads_dir.exists():
            print(f"❌ Upanishads directory not found")
            return []
        
        documents = []
        for upanishad_file in upanishads_dir.glob("*_upanishad.txt"):
            with open(upanishad_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title and focus
            lines = content.split('\n')
            title = lines[0] if lines else upanishad_file.stem
            
            # Split into sections for better retrieval
            sections = content.split('\n\n')
            for i, section in enumerate(sections):
                if section.strip() and len(section) > 100:
                    documents.append({
                        'text': section.strip(),
                        'metadata': {
                            'source': title,
                            'file': upanishad_file.name,
                            'section': i + 1,
                            'category': 'upanishad',
                            'type': 'scripture'
                        }
                    })
        
        print(f"✅ Loaded {len(documents)} Upanishad sections")
        return documents
    
    def load_techniques(self) -> List[Dict[str, Any]]:
        """Load meditation techniques"""
        print("\n🧘 Loading meditation techniques...")
        
        techniques_file = (self.knowledge_dir / "techniques" / 
                          "meditation_techniques.json")
        
        if not techniques_file.exists():
            print(f"❌ Techniques file not found")
            return []
        
        with open(techniques_file, 'r', encoding='utf-8') as f:
            techniques = json.load(f)
        
        documents = []
        for key, technique in techniques.items():
            # Create comprehensive technique document
            doc_text = f"""
Meditation Technique: {technique['name']}
Category: {technique['category']}
Difficulty: {technique['difficulty']}
Duration: {technique['duration']}

Description: {technique['description']}

"""
            
            if 'steps' in technique:
                doc_text += "Steps:\n"
                doc_text += "\n".join(technique['steps'])
                doc_text += "\n\n"
            
            if 'benefits' in technique:
                doc_text += "Benefits:\n"
                doc_text += "\n".join(f"• {b}" for b in technique['benefits'])
                doc_text += "\n\n"
            
            if 'philosophy' in technique:
                doc_text += f"Philosophy: {technique['philosophy']}\n"
            
            documents.append({
                'text': doc_text.strip(),
                'metadata': {
                    'source': technique['name'],
                    'technique_id': key,
                    'category': technique['category'],
                    'difficulty': technique['difficulty'],
                    'type': 'technique'
                }
            })
        
        print(f"✅ Loaded {len(documents)} meditation techniques")
        return documents
    
    def create_embeddings(self):
        """Create embeddings and store in vector database"""
        if not self.embedder or not self.collection:
            print("❌ Cannot create embeddings - missing dependencies")
            return False
        
        print("\n🔄 Creating vector embeddings...")
        print("=" * 60)
        
        # Load all documents
        all_documents = []
        all_documents.extend(self.load_yoga_sutras())
        all_documents.extend(self.load_upanishads())
        all_documents.extend(self.load_techniques())
        
        if not all_documents:
            print("❌ No documents to embed")
            return False
        
        print(f"\n📊 Total documents to embed: {len(all_documents)}")
        print("🔄 Generating embeddings (this may take a minute)...")
        
        # Batch process for efficiency
        batch_size = 50
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            
            # Prepare batch data
            texts = [doc['text'] for doc in batch]
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            metadatas = [doc['metadata'] for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedder.encode(texts, 
                                             convert_to_numpy=True).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"  ✅ Processed batch {i//batch_size + 1}/" 
                  f"{(len(all_documents)-1)//batch_size + 1}")
        
        print(f"\n✅ Created embeddings for {len(all_documents)} documents")
        print(f"✅ Vector database saved at {self.db_path}")
        return True
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the knowledge base"""
        if not self.embedder or not self.collection:
            return {
                'error': 'RAG system not initialized',
                'results': []
            }
        
        # Generate query embedding
        query_embedding = self.embedder.encode(
            question, 
            convert_to_numpy=True
        ).tolist()
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return {
            'query': question,
            'num_results': len(formatted_results),
            'results': formatted_results
        }
    
    def test_queries(self):
        """Test the RAG system with sample queries"""
        print("\n" + "=" * 60)
        print("🧪 TESTING ATRI RAG SYSTEM")
        print("=" * 60)
        
        test_questions = [
            "What is yoga according to Patanjali?",
            "How do I begin meditation practice?",
            "What are the eight limbs of yoga?",
            "Explain So'ham meditation",
            "What is samadhi?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 60)
            
            results = self.query(question, n_results=2)
            
            for i, result in enumerate(results['results'], 1):
                print(f"\n📖 Result {i} (Source: {result['metadata']['source']}):")
                # Show first 200 characters
                text_preview = result['text'][:200] + "..."
                print(f"{text_preview}")
            
            print()


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("🧘 ATRI RAG SYSTEM BUILDER")
    print("Creating vector database for meditation knowledge")
    print("=" * 60)
    
    # Check dependencies
    if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ Missing required dependencies!")
        print("\nInstall with:")
        print("  pip install chromadb sentence-transformers")
        return
    
    # Initialize RAG system
    rag = AtriKnowledgeRAG()
    
    # Create embeddings
    success = rag.create_embeddings()
    
    if not success:
        print("\n❌ Failed to create embeddings")
        return
    
    # Test the system
    rag.test_queries()
    
    # Save metadata
    metadata_file = Path("engines/rishi/rag_systems/atri_metadata.json")
    metadata = {
        "rishi": "Atri",
        "domain": "Meditation and Contemplation",
        "status": "active",
        "created": "2025-10-04",
        "knowledge_sources": {
            "primary_texts": ["Patanjali Yoga Sutras"],
            "upanishads": [
                "Mandukya", "Katha", "Isha", 
                "Svetasvatara", "Kaivalya"
            ],
            "techniques": [
                "Anapanasati", "So'ham", "Trataka", 
                "Vipassana", "Chakra Meditation"
            ]
        },
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "ChromaDB",
        "collection_name": "atri_meditation_knowledge"
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ ATRI RAG SYSTEM CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nMetadata saved: {metadata_file}")
    print("\nNext steps:")
    print("1. Integrate RAG with Atri personality engine")
    print("2. Test with real queries")
    print("3. Fine-tune response generation")
    print()


if __name__ == "__main__":
    main()
