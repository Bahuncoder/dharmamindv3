#!/usr/bin/env python3
"""
Bhrigu RAG System Builder
==========================

Creates vector embeddings and RAG system for Rishi Bhrigu's astrology knowledge.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class BhriguRAGBuilder:
    """Build RAG system for Bhrigu's astrology knowledge"""
    
    def __init__(self):
        self.base_path = Path("data/rishi_knowledge/bhrigu")
        self.rag_path = Path("engines/rishi/rag_systems")
        self.rag_path.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.rag_path / "bhrigu_vector_db")
        )
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="bhrigu_astrology",
            metadata={"description": "Bhrigu's Vedic astrology knowledge"}
        )
        print("✅ Vector database initialized")
        
        self.documents = []
        self.metadatas = []
        self.ids = []
    
    def load_all_knowledge(self):
        """Load all Bhrigu's knowledge documents"""
        print("\n📖 Loading knowledge base...")
        
        self.load_astrology_fundamentals()
        self.load_nakshatras()
        self.load_planets()
        self.load_birth_chart_guide()
        self.load_dasha_system()
        
        print(f"✅ Loaded {len(self.documents)} documents")
    
    def load_astrology_fundamentals(self):
        """Load Vedic astrology fundamentals"""
        json_path = self.base_path / "vedic_astrology" / "astrology_fundamentals.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add each fundamental concept
            for item in data['fundamentals']:
                text = f"{item['concept']}\n"
                if 'sanskrit' in item:
                    text += f"Sanskrit: {item['sanskrit']}\n"
                if 'description' in item:
                    text += f"{item['description']}\n"
                if 'purpose' in item:
                    text += f"Purpose: {item['purpose']}\n"
                if 'pillars' in item:
                    text += "Pillars:\n"
                    for key, val in item['pillars'].items():
                        text += f"  {key}: {val}\n"
                
                self.documents.append(text)
                self.metadatas.append({
                    'type': 'fundamental',
                    'source': 'Vedic Astrology Fundamentals',
                    'concept': item['concept']
                })
                self.ids.append(f"fundamental_{len(self.documents)}")
            
            # Add core teachings
            for teaching in data['core_teachings']:
                self.documents.append(f"Core Teaching: {teaching}")
                self.metadatas.append({
                    'type': 'teaching',
                    'source': 'Vedic Astrology Fundamentals'
                })
                self.ids.append(f"teaching_{len(self.documents)}")
            
            print(f"  ✅ Loaded astrology fundamentals")
    
    def load_nakshatras(self):
        """Load Nakshatras database"""
        json_path = self.base_path / "nakshatras" / "nakshatras_database.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for nak in data['nakshatras']:
                text = f"Nakshatra: {nak['name']} ({nak['sanskrit']})\n"
                text += f"Number: {nak['number']}\n"
                text += f"Range: {nak['range']}\n"
                text += f"Deity: {nak['deity']}\n"
                text += f"Ruling Planet: {nak['ruler']}\n"
                text += f"Symbol: {nak['symbol']}\n"
                text += f"Qualities: {nak['qualities']}\n"
                text += f"Characteristics: {nak['characteristics']}"
                
                self.documents.append(text)
                self.metadatas.append({
                    'type': 'nakshatra',
                    'source': '27 Nakshatras',
                    'name': nak['name'],
                    'number': nak['number']
                })
                self.ids.append(f"nakshatra_{nak['number']}")
            
            print(f"  ✅ Loaded {len(data['nakshatras'])} Nakshatras")
    
    def load_planets(self):
        """Load planetary wisdom"""
        json_path = self.base_path / "planetary_wisdom" / "nine_planets.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for planet in data['planets']:
                text = f"Planet: {planet['name']} - {planet['sanskrit']}\n"
                text += f"Represents: {planet['represents']}\n"
                if 'deity' in planet:
                    text += f"Deity: {planet['deity']}\n"
                if 'exalted_in' in planet:
                    text += f"Exalted in: {planet['exalted_in']}\n"
                    text += f"Debilitated in: {planet['debilitated_in']}\n"
                text += f"Positive traits: {planet['positive_traits']}\n"
                text += f"Negative traits: {planet['negative_traits']}\n"
                text += f"Bhrigu's Teaching: {planet['bhrigu_teaching']}"
                
                self.documents.append(text)
                self.metadatas.append({
                    'type': 'planet',
                    'source': 'Nine Grahas',
                    'name': planet['name']
                })
                self.ids.append(f"planet_{planet['name'].split()[0].lower()}")
            
            print(f"  ✅ Loaded {len(data['planets'])} planets")
    
    def load_birth_chart_guide(self):
        """Load birth chart interpretation guide"""
        text_path = self.base_path / "birth_charts" / "chart_interpretation_guide.txt"
        
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into sections
            sections = content.split('\n\n\n')
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:  # Skip very short sections
                    self.documents.append(section.strip())
                    self.metadatas.append({
                        'type': 'chart_guide',
                        'source': 'Birth Chart Interpretation',
                        'section': i + 1
                    })
                    self.ids.append(f"chart_guide_{i + 1}")
            
            print(f"  ✅ Loaded birth chart guide ({len(sections)} sections)")
    
    def load_dasha_system(self):
        """Load Dasha system guide"""
        text_path = self.base_path / "vedic_astrology" / "dasha_system_guide.txt"
        
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into sections
            sections = content.split('\n\n\n')
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:
                    self.documents.append(section.strip())
                    self.metadatas.append({
                        'type': 'dasha',
                        'source': 'Dasha System',
                        'section': i + 1
                    })
                    self.ids.append(f"dasha_{i + 1}")
            
            print(f"  ✅ Loaded Dasha system ({len(sections)} sections)")
    
    def create_embeddings(self):
        """Generate embeddings and store in vector database"""
        print("\n🔄 Generating embeddings...")
        
        # Generate embeddings
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        # Add to ChromaDB
        self.collection.add(
            documents=self.documents,
            embeddings=embeddings.tolist(),
            metadatas=self.metadatas,
            ids=self.ids
        )
        
        print(f"✅ Created embeddings for {len(self.documents)} documents")
        
        # Save metadata
        metadata = {
            'total_documents': len(self.documents),
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dimensions': 384,
            'knowledge_areas': {
                'fundamentals': sum(1 for m in self.metadatas if m['type'] == 'fundamental'),
                'nakshatras': sum(1 for m in self.metadatas if m['type'] == 'nakshatra'),
                'planets': sum(1 for m in self.metadatas if m['type'] == 'planet'),
                'chart_guide': sum(1 for m in self.metadatas if m['type'] == 'chart_guide'),
                'dasha': sum(1 for m in self.metadatas if m['type'] == 'dasha'),
                'teachings': sum(1 for m in self.metadatas if m['type'] == 'teaching')
            }
        }
        
        metadata_path = self.rag_path / "bhrigu_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Metadata saved to: {metadata_path}")
    
    def query(self, question, n_results=3):
        """Query the RAG system"""
        # Generate embedding for question
        query_embedding = self.model.encode([question])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return {'results': formatted_results, 'query': question}
    
    def test_rag(self):
        """Test the RAG system with sample queries"""
        print("\n" + "="*70)
        print("🔮 TESTING BHRIGU'S RAG SYSTEM")
        print("="*70 + "\n")
        
        test_queries = [
            "What is Vedic astrology?",
            "Explain the Nakshatras",
            "What does Jupiter represent?",
            "How do I read a birth chart?",
            "What is the Dasha system?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"❓ Query: {query}")
            print("="*70)
            
            results = self.query(query, n_results=2)
            
            for i, result in enumerate(results['results'], 1):
                print(f"\n{i}. From {result['metadata']['source']}:")
                print(f"   Type: {result['metadata']['type']}")
                print(f"   Text: {result['text'][:200]}...")
                if result['distance']:
                    print(f"   Relevance: {1 - result['distance']:.2%}")
            
            print()


class BhriguKnowledgeRAG:
    """Simple interface to query Bhrigu's knowledge"""
    
    def __init__(self):
        self.rag_path = Path("engines/rishi/rag_systems")
        
        # Load model
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Load ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.rag_path / "bhrigu_vector_db")
        )
        self.collection = self.client.get_collection(name="bhrigu_astrology")
        print(f"✅ Vector database initialized at {self.rag_path / 'bhrigu_vector_db'}")
    
    def query(self, question, n_results=3):
        """Query Bhrigu's knowledge"""
        query_embedding = self.model.encode([question])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return {'results': formatted_results, 'query': question}


def main():
    """Build and test Bhrigu's RAG system"""
    print("\n" + "="*70)
    print("🔮 BUILDING BHRIGU'S RAG SYSTEM")
    print("="*70 + "\n")
    
    # Build RAG
    builder = BhriguRAGBuilder()
    builder.load_all_knowledge()
    builder.create_embeddings()
    
    # Test RAG
    builder.test_rag()
    
    print("\n" + "="*70)
    print("✅ BHRIGU'S RAG SYSTEM COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
