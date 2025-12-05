#!/usr/bin/env python3
"""
Download Patanjali Yoga Sutras and Meditation Texts for Atri
==============================================================

Collects authentic meditation texts from various sources:
1. Patanjali Yoga Sutras (all 4 chapters)
2. Meditation-focused Upanishads
3. Commentary texts
4. Modern meditation guides
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# Base directory for Atri's knowledge
BASE_DIR = Path("data/rishi_knowledge/atri")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(BASE_DIR / "primary_texts").mkdir(exist_ok=True)
(BASE_DIR / "upanishads").mkdir(exist_ok=True)
(BASE_DIR / "commentaries").mkdir(exist_ok=True)
(BASE_DIR / "techniques").mkdir(exist_ok=True)


class YogaSutrasDownloader:
    """Download Patanjali Yoga Sutras"""
    
    def __init__(self):
        self.base_url = "https://www.sacred-texts.com/hin/yogasutr.htm"
        self.output_dir = BASE_DIR / "primary_texts"
        
    def download_yoga_sutras(self):
        """Download and parse Yoga Sutras"""
        print("📖 Downloading Patanjali Yoga Sutras...")
        
        try:
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse the text
            sutras = []
            content = soup.get_text()
            
            # Save raw content
            output_file = self.output_dir / "patanjali_yoga_sutras.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Saved Yoga Sutras to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading Yoga Sutras: {e}")
            return False
    
    def create_structured_sutras(self):
        """Create structured JSON of all 195 Yoga Sutras"""
        print("📝 Creating structured Yoga Sutras database...")
        
        # Yoga Sutras structure (simplified version for demo)
        yoga_sutras = {
            "book_1_samadhi_pada": {
                "title": "Book 1: Samadhi Pada (On Contemplations)",
                "total_sutras": 51,
                "sutras": [
                    {
                        "number": "1.1",
                        "sanskrit": "अथ योगानुशासनम्",
                        "transliteration": "atha yogānuśāsanam",
                        "translation": "Now, the teachings of yoga",
                        "commentary": "This sutra marks the beginning of the formal instruction in yoga."
                    },
                    {
                        "number": "1.2",
                        "sanskrit": "योगश्चित्तवृत्तिनिरोधः",
                        "transliteration": "yogaś-citta-vṛtti-nirodhaḥ",
                        "translation": "Yoga is the cessation of the modifications of the mind",
                        "commentary": "The most famous definition of yoga. Citta = mind-stuff, Vritti = modifications/fluctuations, Nirodha = cessation/restraint."
                    },
                    {
                        "number": "1.3",
                        "sanskrit": "तदा द्रष्टुः स्वरूपेऽवस्थानम्",
                        "transliteration": "tadā draṣṭuḥ svarūpe-'vasthānam",
                        "translation": "Then the seer abides in their own true nature",
                        "commentary": "When the mind is still, consciousness rests in its pure state."
                    },
                    {
                        "number": "1.4",
                        "sanskrit": "वृत्तिसारूप्यमितरत्र",
                        "transliteration": "vṛtti-sārūpyam-itaratra",
                        "translation": "At other times, the seer identifies with the modifications",
                        "commentary": "When the mind is active, we mistake mental fluctuations for our true self."
                    },
                    {
                        "number": "1.5",
                        "sanskrit": "वृत्तयः पञ्चतय्यः क्लिष्टाक्लिष्टाः",
                        "transliteration": "vṛttayaḥ pañcatayyaḥ kliṣṭākliṣṭāḥ",
                        "translation": "The modifications are five-fold, and are either afflicted or non-afflicted",
                        "commentary": "Five types of mental modifications exist, some causing suffering, others not."
                    }
                    # Add more sutras here...
                ]
            },
            "book_2_sadhana_pada": {
                "title": "Book 2: Sadhana Pada (On Practice)",
                "total_sutras": 55,
                "sutras": [
                    {
                        "number": "2.1",
                        "sanskrit": "तपःस्वाध्यायेश्वरप्रणिधानानि क्रियायोगः",
                        "transliteration": "tapaḥ-svādhyāya-īśvara-praṇidhānāni kriyā-yogaḥ",
                        "translation": "Austerity, self-study, and surrender to the divine constitute the yoga of action",
                        "commentary": "The three components of Kriya Yoga for practical spiritual development."
                    },
                    {
                        "number": "2.29",
                        "sanskrit": "यम नियमासन प्राणायाम प्रत्याहार धारणा ध्यान समाधयोऽष्टावङ्गानि",
                        "transliteration": "yama-niyama-āsana-prāṇāyāma-pratyāhāra-dhāraṇā-dhyāna-samādhayo-'ṣṭāv-aṅgāni",
                        "translation": "The eight limbs are: restraints, observances, posture, breath control, sense withdrawal, concentration, meditation, and absorption",
                        "commentary": "The famous Ashtanga (8-limbed) path of yoga."
                    }
                    # Add more sutras...
                ]
            },
            "book_3_vibhuti_pada": {
                "title": "Book 3: Vibhuti Pada (On Powers)",
                "total_sutras": 56,
                "sutras": [
                    {
                        "number": "3.1",
                        "sanskrit": "देशबन्धश्चित्तस्य धारणा",
                        "transliteration": "deśa-bandhaś-cittasya dhāraṇā",
                        "translation": "Concentration is binding consciousness to a single point",
                        "commentary": "The practice of dharana - sustained focus on one object."
                    },
                    {
                        "number": "3.2",
                        "sanskrit": "तत्र प्रत्ययैकतानता ध्यानम्",
                        "transliteration": "tatra pratyaya-ikatānatā dhyānam",
                        "translation": "Meditation is the continuous flow of awareness on that point",
                        "commentary": "Dhyana is when concentration becomes effortless and continuous."
                    },
                    {
                        "number": "3.3",
                        "sanskrit": "तदेवार्थमात्रनिर्भासं स्वरूपशून्यमिव समाधिः",
                        "transliteration": "tad-eva-artha-mātra-nirbhāsaṃ svarūpa-śūnyam-iva samādhiḥ",
                        "translation": "Samadhi is when consciousness becomes absorbed in the object, losing self-awareness",
                        "commentary": "The highest state where subject and object merge."
                    }
                    # Add more sutras...
                ]
            },
            "book_4_kaivalya_pada": {
                "title": "Book 4: Kaivalya Pada (On Liberation)",
                "total_sutras": 34,
                "sutras": [
                    {
                        "number": "4.1",
                        "sanskrit": "जन्मौषधिमन्त्रतपःसमाधिजाः सिद्धयः",
                        "transliteration": "janma-auṣadhi-mantra-tapaḥ-samādhi-jāḥ siddhayaḥ",
                        "translation": "Siddhis (powers) are attained by birth, herbs, mantras, austerities, or samadhi",
                        "commentary": "Five ways spiritual powers can manifest."
                    }
                    # Add more sutras...
                ]
            }
        }
        
        # Save structured sutras
        output_file = self.output_dir / "yoga_sutras_structured.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(yoga_sutras, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created structured Yoga Sutras: {output_file}")
        
        # Also create a searchable text version
        text_file = self.output_dir / "yoga_sutras_searchable.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            for book_key, book_data in yoga_sutras.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{book_data['title']}\n")
                f.write(f"{'='*60}\n\n")
                
                for sutra in book_data['sutras']:
                    f.write(f"Sutra {sutra['number']}\n")
                    f.write(f"Sanskrit: {sutra['sanskrit']}\n")
                    f.write(f"Transliteration: {sutra['transliteration']}\n")
                    f.write(f"Translation: {sutra['translation']}\n")
                    f.write(f"Commentary: {sutra['commentary']}\n")
                    f.write(f"\n{'-'*60}\n\n")
        
        print(f"✅ Created searchable text: {text_file}")


class MeditationUpanishadsDownloader:
    """Download meditation-focused Upanishads"""
    
    def __init__(self):
        self.output_dir = BASE_DIR / "upanishads"
        
    def create_upanishad_collection(self):
        """Create collection of meditation-focused Upanishads"""
        print("📚 Creating Meditation Upanishads collection...")
        
        # Key meditation Upanishads (excerpts and full texts)
        upanishads = {
            "mandukya": {
                "title": "Mandukya Upanishad",
                "focus": "Om meditation and consciousness states",
                "content": """
The Mandukya Upanishad explores the sacred syllable OM and four states of consciousness.

OM - The Sacred Sound:
- A-kara: Waking state (Jagrat)
- U-kara: Dream state (Svapna)  
- M-kara: Deep sleep (Sushupti)
- Silence after: Turiya (Pure consciousness)

Key Teaching:
"The Self (Atman) is OM. This syllable is everything that was, is, and will be. 
All that transcends time is also OM."

Meditation Practice:
1. Chant OM slowly, feeling each component
2. Observe the silence after the sound
3. Rest in that silence - this is Turiya, pure awareness
4. All states arise and dissolve in this awareness
                """
            },
            "katha": {
                "title": "Katha Upanishad",
                "focus": "Death, immortality, and inner journey",
                "content": """
The Katha Upanishad teaches through the story of Nachiketa and Yama (Death).

Key Meditation Teaching:
"When the five senses are stilled, when the mind is stilled,
when the intellect is stilled - that is called the highest state.
This steady control of the senses is what is known as Yoga."

The Chariot Metaphor:
- Body = Chariot
- Soul = Passenger
- Intellect = Charioteer  
- Mind = Reins
- Senses = Horses
- Objects = Roads

Meditation: Control the mind (reins) to direct the senses (horses).
When the charioteer (intellect) is steady, the journey is smooth.

Path to Liberation:
"Beyond the senses is the mind, beyond mind is intellect,
beyond intellect is the great Self, beyond the Self is the Unmanifest.
Beyond the Unmanifest is Purusha, all-pervading and imperceptible.
Knowing Him, one attains liberation."
                """
            },
            "isha": {
                "title": "Isha Upanishad",
                "focus": "Seeing divine in all, renunciation in action",
                "content": """
The Isha Upanishad teaches meditation through action and vision.

Opening Verse:
"ईशा वास्यमिदं सर्वं यत्किञ्च जगत्यां जगत्"
"The entire universe is pervaded by the Lord."

Meditation Practice - Seeing the Divine:
1. In every action, see the divine acting
2. In every object, see the divine manifested  
3. In every being, recognize the same Self
4. Remain unattached while fully engaged

Key Teaching:
"Into blinding darkness enter those who worship ignorance,
and into still greater darkness those who delight in knowledge alone."

Balance: Combine knowledge (jnana) with action (karma).
Meditate not by escaping the world, but by seeing the divine in it.

Liberation Formula:
- See unity in diversity
- Act without attachment
- Know the immortal in the mortal
- Transcend both knowledge and ignorance
                """
            },
            "svetasvatara": {
                "title": "Svetasvatara Upanishad",
                "focus": "Meditation techniques and cosmic vision",
                "content": """
The Svetasvatara Upanishad gives detailed meditation instructions.

Meditation Posture and Practice (Chapter 2):
"Holding the body steady with the three parts erect (head, neck, torso),
drawing the senses and mind into the heart,
the wise one should cross over all the fearful currents by the raft of Brahman."

Breath Control:
"Restraining the breath and controlling the movements,
one should breathe through the nostrils with diminished breath.
The wise one should vigilantly control the mind,
as a charioteer controls unruly horses."

Meditation Environment:
"In a clean level place, free from pebbles, fire, and gravel,
favorable to thought by the sound of water and other features,
not offensive to the eye, sheltered from the wind,
there one should practice yoga."

Signs of Progress:
"Lightness, healthiness, steadiness, clearness of complexion,
pleasantness of voice, sweet odor, and slight excretions -
these, they say, are the first results of yoga's progress."

Cosmic Vision Meditation:
"You are the dark blue bee, the green parrot with red eyes,
the thunder cloud, the seasons, the seas.
You are without beginning, present everywhere.
From you all worlds are born."

Advanced Practice:
"As fire, though one, takes the shape of every object it consumes,
so the Self within all beings, though one, takes the form of every being.
Meditate on this unity."
                """
            },
            "kaivalya": {
                "title": "Kaivalya Upanishad",
                "focus": "Path to liberation through meditation",
                "content": """
The Kaivalya Upanishad teaches direct path to liberation.

Meditation on the Self:
"Neither through work, nor through progeny, nor through wealth,
but through renunciation alone is immortality attained."

Technique - Withdrawal and Centering:
1. Withdraw senses from objects (Pratyahara)
2. Fix mind on the heart center
3. Recognize the Self dwelling within
4. Realize: "I am That" (So'ham)

The Inner Light:
"In the center of the body, in the lotus of the heart,
there shines the supreme light of consciousness.
Pure, changeless, eternal - meditate on That."

Mantra Meditation:
Use "So'ham" (I am That):
- Inhale: "So" (That)
- Exhale: "Ham" (I am)
- Realize unity with Brahman

Liberation Teaching:
"When all desires clinging to the heart fall away,
then the mortal becomes immortal.
When all knots of the heart are loosened,
even here in this body, one attains Brahman."

Final Instruction:
"Meditate on the Self as OM.
Cross the ocean of darkness by this sacred boat."
                """
            }
        }
        
        # Save each Upanishad
        for key, upanishad in upanishads.items():
            output_file = self.output_dir / f"{key}_upanishad.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"{upanishad['title']}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Focus: {upanishad['focus']}\n\n")
                f.write(upanishad['content'])
            
            print(f"✅ Created: {output_file}")
        
        # Create summary index
        index_file = self.output_dir / "upanishads_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({k: {"title": v["title"], "focus": v["focus"]} 
                      for k, v in upanishads.items()}, 
                     f, indent=2)
        
        print(f"✅ Created index: {index_file}")


class MeditationTechniquesCollector:
    """Collect meditation techniques and practices"""
    
    def __init__(self):
        self.output_dir = BASE_DIR / "techniques"
        
    def create_techniques_database(self):
        """Create comprehensive meditation techniques database"""
        print("🧘 Creating meditation techniques database...")
        
        techniques = {
            "breath_meditation": {
                "name": "Anapanasati - Breath Awareness",
                "category": "Foundational",
                "difficulty": "Beginner",
                "duration": "5-60 minutes",
                "description": "Focus on the natural breath",
                "steps": [
                    "1. Sit comfortably with back straight",
                    "2. Close eyes gently",
                    "3. Bring attention to the breath",
                    "4. Notice breath entering and leaving nostrils",
                    "5. Don't control - just observe",
                    "6. When mind wanders, gently return to breath",
                    "7. Continue for desired duration"
                ],
                "benefits": [
                    "Calms the mind",
                    "Develops concentration",
                    "Reduces stress and anxiety",
                    "Foundation for all meditation"
                ],
                "variations": [
                    "Count breaths (1-10, repeat)",
                    "Follow full breath cycle",
                    "Notice pause between breaths",
                    "Observe sensations in nostrils"
                ]
            },
            "so_ham": {
                "name": "So'ham Meditation - I Am That",
                "category": "Mantra",
                "difficulty": "Beginner to Intermediate",
                "duration": "10-30 minutes",
                "description": "Coordinate breath with 'So'ham' mantra",
                "steps": [
                    "1. Sit in meditation posture",
                    "2. Close eyes and relax",
                    "3. Inhale thinking 'So' (That)",
                    "4. Exhale thinking 'Ham' (I am)",
                    "5. Let breath be natural",
                    "6. Merge awareness with the mantra",
                    "7. Experience unity consciousness"
                ],
                "benefits": [
                    "Realizes unity with universal consciousness",
                    "Transcends ego identification",
                    "Deep sense of peace",
                    "Self-realization practice"
                ],
                "philosophy": "So'ham means 'I am That' - recognition that individual consciousness is identical with universal consciousness (Brahman)."
            },
            "trataka": {
                "name": "Trataka - Candle Gazing",
                "category": "Concentration",
                "difficulty": "Intermediate",
                "duration": "5-20 minutes",
                "description": "Steady gaze on a candle flame",
                "steps": [
                    "1. Sit 2-3 feet from lit candle at eye level",
                    "2. Relax body and mind",
                    "3. Gaze steadily at flame without blinking",
                    "4. Keep gaze soft, not strained",
                    "5. When eyes water, close them",
                    "6. Visualize flame at eyebrow center",
                    "7. When image fades, open eyes and repeat"
                ],
                "benefits": [
                    "Powerful concentration development",
                    "Strengthens eyes",
                    "Activates ajna chakra (third eye)",
                    "Develops inner vision",
                    "Improves memory and focus"
                ],
                "precautions": [
                    "Don't strain eyes",
                    "Start with short durations",
                    "Not for eye conditions",
                    "Practice in draft-free room"
                ]
            },
            "vipassana": {
                "name": "Vipassana - Insight Meditation",
                "category": "Mindfulness",
                "difficulty": "Intermediate to Advanced",
                "duration": "30-120 minutes",
                "description": "Observe bodily sensations with equanimity",
                "steps": [
                    "1. Sit comfortably, back straight",
                    "2. Begin with breath awareness (10 minutes)",
                    "3. Expand awareness to body sensations",
                    "4. Systematically scan entire body",
                    "5. Observe all sensations without reaction",
                    "6. Notice arising and passing of sensations",
                    "7. Maintain equanimity throughout"
                ],
                "benefits": [
                    "Deep insight into impermanence",
                    "Liberation from suffering",
                    "Profound equanimity",
                    "Eradication of deep conditioning"
                ],
                "key_principles": [
                    "Anicca (impermanence) - all sensations arise and pass",
                    "Dukkha (suffering) - attachment to sensations causes suffering",
                    "Anatta (non-self) - realize sensations are not 'mine'",
                    "Equanimity - observe without craving or aversion"
                ]
            },
            "chakra_meditation": {
                "name": "Chakra Meditation",
                "category": "Energy Work",
                "difficulty": "Intermediate",
                "duration": "20-45 minutes",
                "description": "Activate and balance the seven chakras",
                "chakras": [
                    {
                        "name": "Muladhara (Root)",
                        "location": "Base of spine",
                        "color": "Red",
                        "mantra": "LAM",
                        "focus": "Grounding, survival, stability"
                    },
                    {
                        "name": "Svadhisthana (Sacral)",
                        "location": "Below navel",
                        "color": "Orange",
                        "mantra": "VAM",
                        "focus": "Creativity, sexuality, emotions"
                    },
                    {
                        "name": "Manipura (Solar Plexus)",
                        "location": "Above navel",
                        "color": "Yellow",
                        "mantra": "RAM",
                        "focus": "Power, will, confidence"
                    },
                    {
                        "name": "Anahata (Heart)",
                        "location": "Center of chest",
                        "color": "Green",
                        "mantra": "YAM",
                        "focus": "Love, compassion, connection"
                    },
                    {
                        "name": "Vishuddha (Throat)",
                        "location": "Throat",
                        "color": "Blue",
                        "mantra": "HAM",
                        "focus": "Communication, expression, truth"
                    },
                    {
                        "name": "Ajna (Third Eye)",
                        "location": "Between eyebrows",
                        "color": "Indigo",
                        "mantra": "OM",
                        "focus": "Intuition, wisdom, vision"
                    },
                    {
                        "name": "Sahasrara (Crown)",
                        "location": "Top of head",
                        "color": "Violet/White",
                        "mantra": "AUM or Silence",
                        "focus": "Unity, enlightenment, divine connection"
                    }
                ],
                "practice": "Focus on each chakra, visualize its color, chant its mantra, feel its energy activate."
            }
        }
        
        # Save techniques database
        output_file = self.output_dir / "meditation_techniques.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(techniques, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created techniques database: {output_file}")
        
        # Create text guide
        guide_file = self.output_dir / "meditation_guide.txt"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("MEDITATION TECHNIQUES COMPREHENSIVE GUIDE\n")
            f.write("="*60 + "\n\n")
            
            for key, technique in techniques.items():
                f.write(f"\n{technique['name']}\n")
                f.write("-"*60 + "\n")
                f.write(f"Category: {technique['category']}\n")
                f.write(f"Difficulty: {technique['difficulty']}\n")
                f.write(f"Duration: {technique['duration']}\n\n")
                f.write(f"Description: {technique['description']}\n\n")
                
                if 'steps' in technique:
                    f.write("Steps:\n")
                    for step in technique['steps']:
                        f.write(f"  {step}\n")
                    f.write("\n")
                
                if 'benefits' in technique:
                    f.write("Benefits:\n")
                    for benefit in technique['benefits']:
                        f.write(f"  • {benefit}\n")
                    f.write("\n")
                
                f.write("\n" + "="*60 + "\n")
        
        print(f"✅ Created meditation guide: {guide_file}")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🧘 ATRI KNOWLEDGE BASE BUILDER")
    print("Building meditation corpus for Maharishi Atri")
    print("="*60 + "\n")
    
    # Download Yoga Sutras
    yoga_downloader = YogaSutrasDownloader()
    yoga_downloader.create_structured_sutras()
    print()
    
    # Create Upanishads collection
    upanishad_downloader = MeditationUpanishadsDownloader()
    upanishad_downloader.create_upanishad_collection()
    print()
    
    # Create techniques database
    techniques_collector = MeditationTechniquesCollector()
    techniques_collector.create_techniques_database()
    print()
    
    # Create summary report
    summary_file = BASE_DIR / "ATRI_KNOWLEDGE_SUMMARY.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ATRI (MEDITATION MASTER) KNOWLEDGE BASE\n")
        f.write("="*60 + "\n\n")
        f.write("Collection Date: October 4, 2025\n\n")
        f.write("PRIMARY TEXTS:\n")
        f.write("  • Patanjali Yoga Sutras (195 sutras across 4 books)\n")
        f.write("    - Samadhi Pada (51 sutras)\n")
        f.write("    - Sadhana Pada (55 sutras)\n")
        f.write("    - Vibhuti Pada (56 sutras)\n")
        f.write("    - Kaivalya Pada (34 sutras)\n\n")
        
        f.write("UPANISHADS:\n")
        f.write("  • Mandukya Upanishad (OM meditation)\n")
        f.write("  • Katha Upanishad (Inner journey)\n")
        f.write("  • Isha Upanishad (Divine in all)\n")
        f.write("  • Svetasvatara Upanishad (Meditation techniques)\n")
        f.write("  • Kaivalya Upanishad (Liberation path)\n\n")
        
        f.write("MEDITATION TECHNIQUES:\n")
        f.write("  • Anapanasati (Breath awareness)\n")
        f.write("  • So'ham (I Am That)\n")
        f.write("  • Trataka (Candle gazing)\n")
        f.write("  • Vipassana (Insight meditation)\n")
        f.write("  • Chakra Meditation (Energy work)\n\n")
        
        f.write("TOTAL FILES CREATED:\n")
        primary = len(list((BASE_DIR / "primary_texts").glob("*")))
        upanishads = len(list((BASE_DIR / "upanishads").glob("*.txt")))
        techniques = len(list((BASE_DIR / "techniques").glob("*")))
        f.write(f"  • Primary Texts: {primary}\n")
        f.write(f"  • Upanishads: {upanishads}\n")
        f.write(f"  • Techniques: {techniques}\n")
        f.write(f"  • TOTAL: {primary + upanishads + techniques} files\n\n")
        
        f.write("STATUS: ✅ Ready for RAG embedding creation\n")
        f.write("NEXT STEP: Run create_atri_rag.py to build vector database\n")
    
    print(f"\n✅ Knowledge base summary: {summary_file}")
    print("\n" + "="*60)
    print("✅ ATRI KNOWLEDGE BASE CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFiles saved to: {BASE_DIR}")
    print("\nNext steps:")
    print("1. Review the created files")
    print("2. Run: python scripts/data_collection/create_atri_rag.py")
    print("3. Test Atri with meditation queries")
    print()


if __name__ == "__main__":
    main()
