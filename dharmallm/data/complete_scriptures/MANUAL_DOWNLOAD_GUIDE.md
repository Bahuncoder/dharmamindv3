
# Manual Download Guide for Complete Hindu Scriptures
## Step-by-Step Instructions


# Manual Download Sources (If Archive.org links don't work)

## Sacred-Texts.com - Complete Texts
- Rig Veda: https://www.sacred-texts.com/hin/rigveda/index.htm
- Vishnu Purana: https://www.sacred-texts.com/hin/vp/index.htm
- Mahabharata: https://www.sacred-texts.com/hin/maha/index.htm
- Ramayana: https://www.sacred-texts.com/hin/rama/index.htm
- Manu Smriti: https://www.sacred-texts.com/hin/manu.htm

## GRETIL - Sanskrit Originals
- All Vedas: http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/1_veda/
- Puranas: http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/3_purana/
- Dharma Sutras: http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/4_rellit/

## Project Gutenberg
- Some Upanishads and Vedic texts available

## Wisdom Library
- Online reading of most texts: https://www.wisdomlib.org/hinduism

---
NOTE: Archive.org URLs above may not all exist exactly as specified.
You may need to search Archive.org for each text manually.


## 🎯 PRIORITY ORDER (Start Here)

### TIER 1: CRITICAL TEXTS (Do First) - ~30 MB
1. **Bhagavad Gita** (1-2 MB)
   - Search Archive.org: "Bhagavad Gita complete English"
   - Or use: https://www.gitasupersite.iitk.ac.in/
   
2. **Bhagavata Purana** (10-15 MB)
   - Search Archive.org: "Srimad Bhagavatam English translation"
   - This is THE most important Purana
   
3. **Vishnu Purana** (3-5 MB)
   - Search Archive.org: "Vishnu Purana H.H. Wilson"
   
4. **Yoga Sutras** (Already downloaded! ✅)

5. **Major Upanishads** (Already have 7! ✅)

### TIER 2: ESSENTIAL VEDAS - ~30 MB
6. **Rig Veda Complete**
   - Search Archive.org: "Rigveda English translation"
   - Or Sacred-Texts: https://www.sacred-texts.com/hin/rigveda/
   
7. **Atharva Veda Complete**
   - Search Archive.org: "Atharva Veda English"

### TIER 3: MORE PURANAS - ~20 MB
8. **Shiva Purana**
9. **Garuda Purana**
10. **Markandeya Purana**

### TIER 4: EPICS (Optional - Very Large)
11. **Mahabharata** (100+ MB - Maybe download key books only)
12. **Ramayana** (8-12 MB)

### TIER 5: SMRITIS
13. **Manu Smriti**
14. **Yajnavalkya Smriti**

---

## 📥 HOW TO DOWNLOAD FROM ARCHIVE.ORG

1. Go to https://archive.org
2. Search for the text (e.g., "Bhagavad Gita English")
3. Find a complete translation version
4. Look for "Download Options" on right side
5. Choose "Plain Text" or "PDF" format
6. Save to: `data/complete_scriptures/<category>/`

Example:
- Save Bhagavad Gita as: `data/complete_scriptures/gita/bhagavad_gita.txt`
- Save Bhagavata Purana as: `data/complete_scriptures/puranas/bhagavata_purana.txt`

---

## 📋 CHECKLIST

Track your downloads:

### Vedas (4 total)
- [ ] Rig Veda Complete (~15 MB)
- [ ] Sama Veda Complete (~3 MB)
- [ ] Yajur Veda Complete (~5 MB)
- [ ] Atharva Veda Complete (~8 MB)

### Puranas (18 major, start with these 8)
- [ ] Vishnu Purana (~3 MB)
- [ ] Shiva Purana (~5 MB)
- [ ] Bhagavata Purana (~10 MB) ⭐ PRIORITY
- [ ] Garuda Purana (~3 MB)
- [ ] Brahma Purana (~2 MB)
- [ ] Markandeya Purana (~2 MB)
- [ ] Kurma Purana (~2 MB)
- [ ] Padma Purana (~4 MB)

### Smritis (Dharma Shastras)
- [ ] Manu Smriti (~1 MB) ⭐ PRIORITY
- [ ] Yajnavalkya Smriti (~800 KB)

### Epics (Itihasas)
- [ ] Ramayana Complete (~8 MB)
- [ ] Mahabharata (Key books only) (~20 MB recommended, not full 100MB)

### Already Downloaded ✅
- [x] Isha Upanishad
- [x] Kena Upanishad
- [x] Katha Upanishad
- [x] Prashna Upanishad
- [x] Mundaka Upanishad
- [x] Taittiriya Upanishad
- [x] Aitareya Upanishad
- [x] Yoga Sutras Complete

---

## 🎯 TARGET CORPUS SIZE

**Goal**: 50-100 MB of complete authentic Hindu scriptures

**Current**: ~2.4 MB (7 Upanishads + Yoga Sutras) ✅
**Needed**: ~47-97 MB more

**Recommended Download Order**:
1. Bhagavad Gita (2 MB) → Total: 4.4 MB
2. Bhagavata Purana (10 MB) → Total: 14.4 MB
3. Vishnu Purana (3 MB) → Total: 17.4 MB
4. Rig Veda (15 MB) → Total: 32.4 MB
5. Manu Smriti (1 MB) → Total: 33.4 MB
6. Ramayana (8 MB) → Total: 41.4 MB
7. Shiva Purana (5 MB) → Total: 46.4 MB
8. Atharva Veda (8 MB) → Total: 54.4 MB ✅ DONE!

---

## 🚀 AFTER DOWNLOADING

Once you have 50+ MB of texts:

```bash
# Process into training format
python data/scripts/process_complete_corpus.py

# This will create:
# - data/training/complete_corpus/train.jsonl
# - data/training/complete_corpus/val.jsonl
# - data/training/complete_corpus/test.jsonl

# Then train the 1.5B model!
```

---

**Remember**: Quality over quantity! Start with Tier 1 (critical texts) first.
The 7 Upanishads and Yoga Sutras you already have are EXCELLENT foundation!
