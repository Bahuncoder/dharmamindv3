#!/usr/bin/env python3
"""
Vashishta RAG System Builder
============================

Creates a Retrieval-Augmented Generation system for Rishi Vashishta's
dharma and ethics knowledge base.

Uses:
- ChromaDB for vector storage
- sentence-transformers for embeddings
- Semantic similarity search for retrieving relevant wisdom
"""

import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VashishtaRAGBuilder:
    """Build RAG system for Vashishta's dharma knowledge"""
    
    def __init__(self):
        self.knowledge_path = Path("data/rishi_knowledge/vashishta")
        self.db_path = Path("engines/rishi/rag_systems/vashishta_vector_db")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        
    def load_documents(self):
        """Load all Vashishta's knowledge documents"""
        print("\n" + "="*70)
        print("📿 Loading Vashishta's Dharma Knowledge")
        print("="*70 + "\n")
        
        # Load dharma fundamentals
        self._load_fundamentals()
        
        # Load Purusharthas
        self._load_purusharthas()
        
        # Load Ashramas
        self._load_ashramas()
        
        # Load ethical dilemmas
        self._load_dilemmas()
        
        # Load modern dharma guide
        self._load_modern_guide()
        
        print(f"\n✅ Loaded {len(self.documents)} documents total")
        return self.documents
    
    def _load_fundamentals(self):
        """Load dharma fundamentals"""
        print("📖 Loading dharma fundamentals...")
        
        json_path = self.knowledge_path / "dharma_shastras" / "dharma_fundamentals.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add overview document
        self.documents.append({
            "content": f"{data['title']}\n\n{data['source']}",
            "metadata": {
                "category": "fundamentals",
                "topic": "overview",
                "source": data['source']
            }
        })
        
        # Add each concept
        for concept in data['core_concepts']:
            content = f"{concept['concept']}\n\n"
            if 'sanskrit' in concept:
                content += f"Sanskrit: {concept['sanskrit']}\n\n"
            if 'definition' in concept:
                content += f"{concept['definition']}\n\n"
            if 'explanation' in concept:
                content += f"{concept['explanation']}\n\n"
            if 'virtues' in concept:
                content += "The Ten Universal Virtues:\n"
                for key, val in concept['virtues'].items():
                    name = key.split('_')[1]
                    content += f"{name}: {val}\n"
                content += f"\n{concept['teaching']}\n"
            if 'principle' in concept:
                content += f"Principle: {concept['principle']}\n"
            
            self.documents.append({
                "content": content,
                "metadata": {
                    "category": "fundamentals",
                    "topic": concept['concept'],
                    "source": data['source']
                }
            })
        
        # Add key principles
        principles_text = "Key Principles of Dharmic Living:\n\n"
        principles_text += "\n".join(f"• {p}" for p in data['key_principles'])
        self.documents.append({
            "content": principles_text,
            "metadata": {
                "category": "fundamentals",
                "topic": "key_principles"
            }
        })
        
        print(f"  ✅ Loaded {len(data['core_concepts']) + 2} fundamental documents")
    
    def _load_purusharthas(self):
        """Load four goals of life"""
        print("🎯 Loading Purusharthas...")
        
        json_path = self.knowledge_path / "dharma_shastras" / "purusharthas.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add overview
        self.documents.append({
            "content": f"{data['title']}\n\n{data['description']}",
            "metadata": {
                "category": "purusharthas",
                "topic": "overview"
            }
        })
        
        # Add each goal
        for goal in data['goals']:
            content = f"{goal['name']}\n\n"
            content += f"{goal['meaning']}\n\n"
            content += f"{goal['description']}\n\n"
            content += f"Importance: {goal['importance']}\n\n"
            content += "How to Pursue:\n"
            content += "\n".join(f"• {step}" for step in goal['how_to_pursue'])
            content += f"\n\nVashishta's Teaching: {goal['vashishta_teaching']}"
            
            self.documents.append({
                "content": content,
                "metadata": {
                    "category": "purusharthas",
                    "topic": goal['name'],
                    "goal": goal['meaning']
                }
            })
        
        print(f"  ✅ Loaded {len(data['goals']) + 1} Purushartha documents")
    
    def _load_ashramas(self):
        """Load four life stages"""
        print("🌱 Loading Ashramas...")
        
        text_path = self.knowledge_path / "life_stages" / "four_ashramas.txt"
        with open(text_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Split by stages
        stages = ["BRAHMACHARYA", "GRIHASTHA", "VANAPRASTHA", "SANNYASA"]
        
        for stage in stages:
            start = full_text.find(f"STAGE")
            if start == -1:
                start = full_text.find(stage)
            
            if start != -1:
                # Find next stage or end
                next_stage_pos = len(full_text)
                for next_stage in stages:
                    if next_stage != stage:
                        pos = full_text.find(next_stage, start + 10)
                        if pos != -1 and pos < next_stage_pos:
                            next_stage_pos = pos
                
                stage_content = full_text[start:next_stage_pos].strip()
                
                self.documents.append({
                    "content": stage_content[:2000],  # Limit length
                    "metadata": {
                        "category": "ashramas",
                        "topic": stage.lower(),
                        "life_stage": stage
                    }
                })
        
        # Add overview
        overview = full_text[:full_text.find("STAGE 1:")].strip()
        self.documents.append({
            "content": overview,
            "metadata": {
                "category": "ashramas",
                "topic": "overview"
            }
        })
        
        print(f"  ✅ Loaded {len(stages) + 1} Ashrama documents")
    
    def _load_dilemmas(self):
        """Load ethical dilemmas"""
        print("⚖️  Loading ethical dilemmas...")
        
        json_path = self.knowledge_path / "ethical_teachings" / "ethical_dilemmas.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add each dilemma
        for dilemma in data['dilemmas']:
            content = f"Dilemma: {dilemma['dilemma']}\n\n"
            content += f"Scenario: {dilemma['scenario']}\n\n"
            content += f"Vashishta's Guidance:\n{dilemma['vashishta_guidance']}\n\n"
            content += f"Principle: {dilemma['principle']}"
            
            self.documents.append({
                "content": content,
                "metadata": {
                    "category": "ethical_dilemmas",
                    "topic": dilemma['dilemma'],
                    "principle": dilemma['principle']
                }
            })
        
        print(f"  ✅ Loaded {len(data['dilemmas'])} ethical dilemma documents")
    
    def _load_modern_guide(self):
        """Load modern dharma guide"""
        print("🌍 Loading modern dharma guide...")
        
        text_path = self.knowledge_path / "modern_dharma" / "modern_dharma_guide.txt"
        with open(text_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Split by main sections
        sections = [
            "WORK AND CAREER DHARMA",
            "RELATIONSHIP DHARMA",
            "TECHNOLOGY AND DHARMA",
            "MONEY AND DHARMA",
            "ENVIRONMENTAL DHARMA",
            "CONFLICT RESOLUTION",
            "VASHISHTA'S DAILY DHARMA CHECKLIST"
        ]
        
        for section in sections:
            start = full_text.find(section)
            if start != -1:
                # Find next section or end
                next_pos = len(full_text)
                for next_section in sections:
                    if next_section != section:
                        pos = full_text.find(next_section, start + 10)
                        if pos != -1 and pos < next_pos:
                            next_pos = pos
                
                section_content = full_text[start:next_pos].strip()
                
                self.documents.append({
                    "content": section_content[:2000],  # Limit length
                    "metadata": {
                        "category": "modern_dharma",
                        "topic": section.lower().replace("'", ""),
                        "section": section
                    }
                })
        
        print(f"  ✅ Loaded {len(sections)} modern dharma documents")
    
    def create_vector_database(self):
        """Create ChromaDB vector database"""
        print("\n" + "="*70)
        print("🔮 Creating Vector Database")
        print("="*70 + "\n")
        
        # Create directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            client.delete_collection("vashishta_dharma")
        except:
            pass
        
        collection = client.create_collection(
            name="vashishta_dharma",
            metadata={"description": "Vashishta's dharma and ethics knowledge"}
        )
        
        # Generate embeddings and add to database
        print("Generating embeddings...")
        contents = [doc['content'] for doc in self.documents]
        embeddings = self.model.encode(contents, show_progress_bar=True)
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=[doc['metadata'] for doc in self.documents],
            ids=[f"doc_{i}" for i in range(len(self.documents))]
        )
        
        print(f"✅ Created vector database with {len(self.documents)} documents")
        
        # Save metadata
        metadata = {
            "total_documents": len(self.documents),
            "categories": list(set(doc['metadata']['category'] for doc in self.documents)),
            "model": "all-MiniLM-L6-v2",
            "embedding_dim": 384
        }
        
        metadata_path = Path("engines/rishi/rag_systems/vashishta_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to {metadata_path}")
        
        return collection
    
    def test_queries(self, collection):
        """Test the RAG system with sample queries"""
        print("\n" + "="*70)
        print("🧪 Testing Vashishta's RAG System")
        print("="*70 + "\n")
        
        test_queries = [
            "What is dharma?",
            "How should I balance work and family?",
            "What are the four goals of life?",
            "Should I forgive someone who hurt me?",
            "How to live ethically in modern times?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            print("-" * 70)
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=2
            )
            
            # Display results
            for j, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                similarity = (1 - distance) * 100
                print(f"\nResult {j} (Relevance: {similarity:.2f}%):")
                print(f"Category: {metadata['category']}")
                print(f"Topic: {metadata['topic']}")
                print(f"Preview: {doc[:200]}...")
        
        print("\n" + "="*70)
        print("✅ All test queries completed successfully!")
        print("="*70 + "\n")


def main():
    """Build and test Vashishta's RAG system"""
    builder = VashishtaRAGBuilder()
    
    # Load documents
    builder.load_documents()
    
    # Create vector database
    collection = builder.create_vector_database()
    
    # Test queries
    builder.test_queries(collection)
    
    print("\n✨ Vashishta's RAG system is ready!")
    print("📿 The ancient sage can now guide seekers on dharma and ethics!")


if __name__ == "__main__":
    main()
