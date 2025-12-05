#!/usr/bin/env python3
"""
Analyze current Sanskrit corpus and show expansion plan
"""
import json
from pathlib import Path

print("=" * 70)
print("📊 CURRENT SANSKRIT CORPUS ANALYSIS")
print("=" * 70)

# Load current corpus
corpus_path = Path("data/sanskrit_original/complete_sanskrit_corpus.json")
with open(corpus_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

meta = data['metadata']
texts = data['texts']

print(f"\n✅ Total Texts: {meta['total_texts']}")
print(f"   Language: {meta['language']}")
print(f"   Script: {meta['script']}")

# Count by category
categories = {}
sources = {}
for text in texts:
    cat = text['category']
    src = text['source']
    categories[cat] = categories.get(cat, 0) + 1
    sources[src] = sources.get(src, 0) + 1

print(f"\n📚 Breakdown by Category:")
for cat, count in sorted(categories.items()):
    print(f"   • {cat}: {count} texts")

print(f"\n📖 Breakdown by Source:")
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"   • {src}: {count} texts")

# Show samples
print(f"\n" + "=" * 70)
print("📝 WHAT WE ALREADY HAVE (Sample)")
print("=" * 70)

print("\n🔥 RIGVEDA (ऋग्वेद) - 8 verses collected:")
rigveda_texts = [t for t in texts if t['category'] == 'veda']
for i, text in enumerate(rigveda_texts[:5], 1):
    print(f"\n{i}. Mandala {text['mandala']}, Sukta {text['sukta']}, Rik {text['rik']}")
    print(f"   Deity: {text.get('deity', 'N/A')}")
    print(f"   Rishi: {text.get('rishi', 'N/A')}")
    print(f"   Sanskrit: {text['sanskrit_original'][:80]}...")

print("\n📿 UPANISHADS (उपनिषद्) - 14 verses collected:")
upanishad_texts = [t for t in texts if t['category'] == 'upanishad']
upanishad_names = {}
for t in upanishad_texts:
    name = t['source']
    upanishad_names[name] = upanishad_names.get(name, 0) + 1

for name, count in sorted(upanishad_names.items()):
    print(f"   • {name}: {count} verse(s)")

print("\n🧘 YOGA SUTRAS (योगसूत्राणि) - 18 sutras collected:")
yoga_texts = [t for t in texts if t['category'] == 'yoga_sutra']
padas = {}
for t in yoga_texts:
    pada = t.get('pada', 'unknown')
    padas[pada] = padas.get(pada, 0) + 1

for pada, count in sorted(padas.items()):
    print(f"   • Pada {pada}: {count} sutra(s)")

print("\n🕉️ BHAGAVAD GITA (भगवद्गीता) - 26 verses collected:")
gita_texts = [t for t in texts if t['category'] == 'gita']
chapters = {}
for t in gita_texts:
    ch = t.get('chapter', 'unknown')
    chapters[ch] = chapters.get(ch, 0) + 1

chapter_list = sorted([ch for ch in chapters.keys() if isinstance(ch, int)])
print(f"   • Chapters covered: {', '.join(map(str, chapter_list))}")
print(f"   • Total verses: {len(gita_texts)}")

# Expansion plan
print(f"\n" + "=" * 70)
print("🚀 EXPANSION PLAN TO 500+ TEXTS")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1. RIGVEDA (ऋग्वेद) - Target: 100+ verses                      │
└─────────────────────────────────────────────────────────────────┘
   Current: 8 verses
   Add: 92+ more verses from:
   
   📖 Mandala 1 (Agni Suktas): 30 verses
      - Complete hymns to Agni (1.1, 1.2, 1.3...)
      - Fire sacrifice mantras
      
   📖 Mandala 2-7 (Family Books): 25 verses
      - Hymns to Indra, Varuna, Mitra
      - Cosmic creation hymns
      
   📖 Mandala 8 (Soma Mandala): 15 verses
      - Soma juice hymns
      - Ritual chants
      
   📖 Mandala 9 (Pavamana): 10 verses
      - Purification hymns
      
   📖 Mandala 10 (Philosophical hymns): 12 verses
      - Nasadiya Sukta (already have)
      - Purusha Sukta (already have)
      - Creation hymns
      - Philosophical verses

┌─────────────────────────────────────────────────────────────────┐
│ 2. UPANISHADS (उपनिषद्) - Target: 150+ verses                   │
└─────────────────────────────────────────────────────────────────┘
   Current: 14 verses from 8 Upanishads
   Add: 136+ more verses:
   
   📖 Isha Upanishad: Complete 18 verses (have 2, add 16)
   📖 Kena Upanishad: Complete 35 verses (have 1, add 34)
   📖 Katha Upanishad: Complete 119 verses (have 2, add 30-40 key)
   📖 Mundaka Upanishad: Complete 64 verses (have 2, add 20-30)
   📖 Mandukya Upanishad: Complete 12 verses (have 1, add 11)
   📖 Chandogya Upanishad: 154 verses (have 2, add 30-40 key)
   📖 Brihadaranyaka: 177 verses (have 2, add 30-40 key)
   📖 Taittiriya Upanishad: Complete 79 verses (have 2, add 20-30)
   📖 Aitareya Upanishad: Add 33 verses (NEW)
   📖 Prashna Upanishad: Add 63 verses (NEW)
   📖 Svetasvatara Upanishad: Add 113 verses (NEW)

┌─────────────────────────────────────────────────────────────────┐
│ 3. YOGA SUTRAS (योगसूत्राणि) - Target: 100+ sutras            │
└─────────────────────────────────────────────────────────────────┘
   Current: 18 sutras across 4 padas
   Add: 82+ more sutras:
   
   📖 Samadhi Pada (51 sutras): Add 44 more
   📖 Sadhana Pada (55 sutras): Add 20-30 key
   📖 Vibhuti Pada (55 sutras): Add 10-15 key
   📖 Kaivalya Pada (34 sutras): Add 8-10 key

┌─────────────────────────────────────────────────────────────────┐
│ 4. BHAGAVAD GITA (भगवद्गीता) - Target: 100+ verses            │
└─────────────────────────────────────────────────────────────────┘
   Current: 26 verses from various chapters
   Add: 74+ more key verses from all 18 chapters:
   
   📖 Chapter 2 (Sankhya Yoga): 10 key verses
   📖 Chapter 3 (Karma Yoga): 8 key verses
   📖 Chapter 4 (Jnana Yoga): 8 key verses
   📖 Chapter 6 (Dhyana Yoga): 8 key verses
   📖 Chapter 9 (Raja-Vidya-Guhya Yoga): 8 key verses
   📖 Chapter 12 (Bhakti Yoga): 8 key verses
   📖 Chapter 15 (Purushottama Yoga): 6 key verses
   📖 Other chapters: 18 key verses

┌─────────────────────────────────────────────────────────────────┐
│ 5. NEW SOURCES - Target: 100+ texts                            │
└─────────────────────────────────────────────────────────────────┘
   
   📖 PURANAS (पुराण):
      • Vishnu Purana: 30 key verses
      • Shiva Purana: 20 key verses
      • Bhagavata Purana: 30 key verses
      
   📖 SURYA SIDDHANTA (सूर्य सिद्धान्त):
      • Astronomy verses: 20 verses
      
   📖 BRAHMA SUTRAS (ब्रह्म सूत्र):
      • Key sutras: 30 verses
      
   📖 DHARMA SHASTRAS (धर्म शास्त्र):
      • Manu Smriti: 20 key verses
      • Yajnavalkya Smriti: 10 verses

┌─────────────────────────────────────────────────────────────────┐
│ 6. VEDIC MANTRAS & STOTRAS - Target: 50+ texts                 │
└─────────────────────────────────────────────────────────────────┘
   
   📖 Gayatri Mantra variations (have basic, add 5 more)
   📖 Mahamrityunjaya Mantra
   📖 Shanti Mantras (10-15)
   📖 Sri Rudram (key passages - 10 verses)
   📖 Chamakam (key passages - 5 verses)
   📖 Vishnu Sahasranama (50 key names with verses)
   📖 Lalita Sahasranama (20 key verses)

""")

print("=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"""
Current Corpus:  {meta['total_texts']} texts
Target Corpus:   500+ texts
Expansion:       {500 - meta['total_texts']}+ new texts needed

Breakdown of 500+ target:
  • Rigveda:           100 verses  (have 8, add 92)
  • Upanishads:        150 verses  (have 14, add 136)
  • Yoga Sutras:       100 sutras  (have 18, add 82)
  • Bhagavad Gita:     100 verses  (have 26, add 74)
  • Puranas:           50 verses   (have 0, add 50)
  • Other Texts:       50+ texts   (have 0, add 50+)
  ─────────────────────────────────────────────────
  TOTAL:              550+ texts

All sources: AUTHENTIC ORIGINAL SANSKRIT (देवनागरी)
No translations, only original texts!
""")

print("\n✅ Ready to collect 500+ authentic Sanskrit texts!")
print("   Next: Run expansion collector script\n")
