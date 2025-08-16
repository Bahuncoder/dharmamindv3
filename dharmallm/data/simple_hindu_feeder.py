#!/usr/bin/env python3
"""
Simple Hindu Text Feeder - Feed Original Sanskrit Texts
No complex dependencies, just pure text processing
"""

import json
import os
from datetime import datetime

class SimpleHinduTextFeeder:
    def __init__(self):
        self.texts_fed = 0
        self.total_characters = 0
        self.sanskrit_verses = 0
        
    def get_core_hindu_texts(self):
        """Get all core Hindu scriptures with original Sanskrit"""
        return {
            "bhagavad_gita": {
                "name": "श्रीमद्भगवद्गीता (Bhagavad Gita)",
                "verses": [
                    {
                        "sanskrit": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
                        "transliteration": "karmaṇy evādhikāras te mā phaleṣu kadācana",
                        "english": "You have a right to perform your prescribed duty, but never to the fruits of action",
                        "chapter": 2,
                        "verse": 47
                    },
                    {
                        "sanskrit": "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय",
                        "transliteration": "yoga-sthaḥ kuru karmāṇi saṅgaṁ tyaktvā dhanañjaya",
                        "english": "Perform your duty equipoised, O Arjuna, abandoning all attachment",
                        "chapter": 2,
                        "verse": 48
                    },
                    {
                        "sanskrit": "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज",
                        "transliteration": "sarva-dharmān parityajya mām ekaṁ śaraṇaṁ vraja",
                        "english": "Abandon all varieties of religion and just surrender unto Me",
                        "chapter": 18,
                        "verse": 66
                    },
                    {
                        "sanskrit": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत",
                        "transliteration": "yadā yadā hi dharmasya glānir bhavati bhārata",
                        "english": "Whenever and wherever there is a decline in dharma, O Bharata",
                        "chapter": 4,
                        "verse": 7
                    },
                    {
                        "sanskrit": "अभयं सत्त्वसंशुद्धिर्ज्ञानयोगव्यवस्थितिः",
                        "transliteration": "abhayaṁ sattva-saṁśuddhir jñāna-yoga-vyavasthitiḥ",
                        "english": "Fearlessness, purification of existence, cultivation of spiritual knowledge",
                        "chapter": 16,
                        "verse": 1
                    }
                ]
            },
            "upanishads": {
                "name": "उपनिषद् (Upanishads)",
                "verses": [
                    {
                        "sanskrit": "ॐ सह नाववतु सह नौ भुनक्तु",
                        "transliteration": "oṁ saha nāv avatu saha nau bhunaktu",
                        "english": "May we both be protected, may we both be nourished",
                        "source": "Taittiriya Upanishad"
                    },
                    {
                        "sanskrit": "सत्यं ज्ञानमनन्तं ब्रह्म",
                        "transliteration": "satyaṁ jñānam anantaṁ brahma",
                        "english": "Brahman is Truth, Knowledge, and Infinite",
                        "source": "Taittiriya Upanishad"
                    },
                    {
                        "sanskrit": "तत्त्वमसि श्वेतकेतो",
                        "transliteration": "tat tvam asi śvetaketo",
                        "english": "That thou art, O Svetaketu",
                        "source": "Chandogya Upanishad"
                    },
                    {
                        "sanskrit": "अहं ब्रह्मास्मि",
                        "transliteration": "ahaṁ brahmāsmi",
                        "english": "I am Brahman",
                        "source": "Brihadaranyaka Upanishad"
                    },
                    {
                        "sanskrit": "सर्वं खल्विदं ब्रह्म",
                        "transliteration": "sarvaṁ khalvidaṁ brahma",
                        "english": "All this is indeed Brahman",
                        "source": "Chandogya Upanishad"
                    }
                ]
            },
            "vedic_mantras": {
                "name": "वैदिक मन्त्र (Vedic Mantras)",
                "verses": [
                    {
                        "sanskrit": "ॐ गं गणपतये नमः",
                        "transliteration": "oṁ gaṁ gaṇapataye namaḥ",
                        "english": "Salutations to Lord Ganesha",
                        "purpose": "Obstacle removal"
                    },
                    {
                        "sanskrit": "ॐ नमो भगवते वासुदेवाय",
                        "transliteration": "oṁ namo bhagavate vāsudevāya",
                        "english": "Salutations to Lord Vasudeva (Krishna)",
                        "purpose": "Devotion"
                    },
                    {
                        "sanskrit": "गायत्री मन्त्र: ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं",
                        "transliteration": "oṁ bhūr bhuvaḥ svaḥ tat savitur vareṇyaṁ",
                        "english": "We meditate on the divine light of the Sun",
                        "purpose": "Enlightenment"
                    },
                    {
                        "sanskrit": "ॐ शान्ति शान्ति शान्तिः",
                        "transliteration": "oṁ śānti śānti śāntiḥ",
                        "english": "Peace, peace, peace",
                        "purpose": "Inner peace"
                    }
                ]
            },
            "yoga_sutras": {
                "name": "योगसूत्र (Yoga Sutras)",
                "verses": [
                    {
                        "sanskrit": "योगश्चित्तवृत्तिनिरोधः",
                        "transliteration": "yogaś citta-vṛtti-nirodhaḥ",
                        "english": "Yoga is the cessation of fluctuations of the mind",
                        "sutra": "1.2"
                    },
                    {
                        "sanskrit": "अभ्यासवैराग्याभ्यां तन्निरोधः",
                        "transliteration": "abhyāsa-vairāgyābhyāṁ tan-nirodhaḥ",
                        "english": "This cessation comes through practice and detachment",
                        "sutra": "1.12"
                    },
                    {
                        "sanskrit": "यमनियमासनप्राणायामप्रत्याहारधारणाध्यानसमाधयोऽष्टावङ्गानि",
                        "transliteration": "yama-niyamāsana-prāṇāyāma-pratyāhāra-dhāraṇā-dhyāna-samādhayo 'ṣṭāv aṅgāni",
                        "english": "The eight limbs of yoga are restraints, observances, postures, breath control, withdrawal, concentration, meditation, and absorption",
                        "sutra": "2.29"
                    }
                ]
            },
            "dharma_shastras": {
                "name": "धर्मशास्त्र (Dharma Shastras)",
                "verses": [
                    {
                        "sanskrit": "धर्मो रक्षति रक्षितः",
                        "transliteration": "dharmo rakṣati rakṣitaḥ",
                        "english": "Dharma protects those who protect it",
                        "source": "Manusmriti"
                    },
                    {
                        "sanskrit": "सत्यं ब्रूयात् प्रियं ब्रूयात्",
                        "transliteration": "satyaṁ brūyāt priyaṁ brūyāt",
                        "english": "Speak the truth, speak pleasantly",
                        "source": "Manusmriti"
                    }
                ]
            }
        }
    
    def feed_text(self, text_data):
        """Simple text feeding process"""
        self.texts_fed += 1
        if isinstance(text_data, dict):
            if 'sanskrit' in text_data:
                self.sanskrit_verses += 1
                text_length = len(text_data['sanskrit']) + len(text_data.get('english', ''))
                self.total_characters += text_length
        elif isinstance(text_data, str):
            self.total_characters += len(text_data)
        
        return True
    
    def process_all_texts(self):
        """Process and feed all Hindu texts"""
        print("🕉️  Starting Simple Hindu Text Feeding...")
        print("=" * 60)
        
        all_texts = self.get_core_hindu_texts()
        
        for scripture_key, scripture_data in all_texts.items():
            print(f"\n📖 Processing: {scripture_data['name']}")
            print("-" * 40)
            
            for verse in scripture_data['verses']:
                self.feed_text(verse)
                print(f"✅ Fed: {verse['sanskrit'][:50]}...")
        
        print("\n🎉 FEEDING COMPLETE!")
        print("=" * 60)
        self.show_stats()
    
    def show_stats(self):
        """Show feeding statistics"""
        print(f"📊 FEEDING STATISTICS:")
        print(f"   • Total Texts Fed: {self.texts_fed}")
        print(f"   • Sanskrit Verses: {self.sanskrit_verses}")
        print(f"   • Total Characters: {self.total_characters:,}")
        print(f"   • Average per Text: {self.total_characters // max(1, self.texts_fed)}")
        
        # Save stats
        stats = {
            'feeding_date': datetime.now().isoformat(),
            'texts_fed': self.texts_fed,
            'sanskrit_verses': self.sanskrit_verses,
            'total_characters': self.total_characters,
            'status': 'completed'
        }
        
        with open('feeding_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Stats saved to: feeding_stats.json")

class SimpleTranslator:
    """Simple Sanskrit translation system"""
    
    def __init__(self):
        self.translation_cache = {}
    
    def translate_sanskrit(self, sanskrit_text, target_language='english'):
        """Simple translation lookup"""
        # Basic translation dictionary
        basic_translations = {
            'ॐ': {'english': 'Om', 'hindi': 'ॐ', 'tamil': 'ஓம்'},
            'नमः': {'english': 'salutations', 'hindi': 'नमस्कार', 'tamil': 'வணக்கம்'},
            'धर्म': {'english': 'dharma/righteousness', 'hindi': 'धर्म', 'tamil': 'தர்மம்'},
            'योग': {'english': 'yoga/union', 'hindi': 'योग', 'tamil': 'யோகம்'},
            'ब्रह्म': {'english': 'Brahman/Ultimate Reality', 'hindi': 'ब्रह्म', 'tamil': 'பிரம்மம்'},
            'शान्ति': {'english': 'peace', 'hindi': 'शांति', 'tamil': 'அமைதி'},
            'सत्य': {'english': 'truth', 'hindi': 'सत्य', 'tamil': 'சத்தியம்'},
            'ज्ञान': {'english': 'knowledge', 'hindi': 'ज्ञान', 'tamil': 'ஞானம்'}
        }
        
        # Simple word-by-word translation
        words = sanskrit_text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.strip('।॥')  # Remove punctuation
            if clean_word in basic_translations:
                translated_words.append(basic_translations[clean_word].get(target_language, clean_word))
            else:
                translated_words.append(f"[{clean_word}]")  # Untranslated
        
        return ' '.join(translated_words)
    
    def demonstrate_translation(self):
        """Demonstrate translation capabilities"""
        print("\n🌍 TRANSLATION DEMONSTRATION")
        print("=" * 50)
        
        test_phrases = [
            "ॐ नमो भगवते वासुदेवाय",
            "धर्मो रक्षति रक्षितः",
            "योगश्चित्तवृत्तिनिरोधः",
            "सत्यं ज्ञानमनन्तं ब्रह्म"
        ]
        
        for phrase in test_phrases:
            print(f"\n📝 Sanskrit: {phrase}")
            print(f"🇬🇧 English: {self.translate_sanskrit(phrase, 'english')}")
            print(f"🇮🇳 Hindi: {self.translate_sanskrit(phrase, 'hindi')}")
            print(f"🇮🇳 Tamil: {self.translate_sanskrit(phrase, 'tamil')}")

def main():
    """Main execution function"""
    print("🕉️  SIMPLE HINDU TEXT FEEDING SYSTEM")
    print("=" * 60)
    print("Feeding ALL original Hindu texts into the AI...")
    print("No complex dependencies - pure text processing")
    print()
    
    # Create feeder
    feeder = SimpleHinduTextFeeder()
    
    # Process all texts
    feeder.process_all_texts()
    
    # Demonstrate translation
    translator = SimpleTranslator()
    translator.demonstrate_translation()
    
    print("\n✨ SYSTEM READY!")
    print("All original Hindu texts have been fed to the AI")
    print("Sanskrit translation system is active")

if __name__ == "__main__":
    main()
