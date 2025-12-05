# 📚 How to Add PDFs and Process Everything

## 🎯 ONE SCRIPT DOES IT ALL!

The `process_complete_corpus.py` script now automatically handles:
- ✅ **PDF files** (complete books from Archive.org)
- ✅ **HTML files** (from Sacred-Texts mass download)
- ✅ **TXT files** (manual downloads)
- ✅ **JSON files** (structured data)

---

## 📥 Step 1: Download PDFs

### Where to Download:
1. **Archive.org** - BEST source for complete Hindu texts
2. **PDF Drive**
3. **Google Books** (use extensions to download full PDFs)
4. **VedaBase.io** - Bhagavata Purana PDFs

### Recommended Downloads (Get GB-scale data!):

#### 🔥 HIGH PRIORITY (Large texts):
```
1. Complete Mahabharata (Ganguli translation)
   → Search: "mahabharata ganguli pdf"
   → Size: ~200 MB
   → Download to: data/complete_scriptures/itihasas/mahabharata.pdf

2. Srimad Bhagavatam (Complete 18 Cantos)
   → Search: "bhagavata purana complete pdf"
   → vedabase.io has complete PDFs
   → Size: ~500 MB
   → Download to: data/complete_scriptures/puranas/bhagavata_purana.pdf

3. All 18 Puranas Collection
   → Search each Purana individually:
     - Vishnu Purana
     - Shiva Purana
     - Brahma Purana
     - Garuda Purana
     - Skanda Purana
     - etc.
   → Size: ~5 GB total
   → Download to: data/complete_scriptures/puranas/*.pdf

4. Complete Ramayana (Valmiki)
   → Search: "valmiki ramayana complete pdf"
   → Size: ~500 MB
   → Download to: data/complete_scriptures/itihasas/ramayana.pdf

5. 108 Upanishads Collection
   → Search: "108 upanishads pdf"
   → Size: ~200 MB
   → Download to: data/complete_scriptures/upanishads/108_upanishads.pdf
```

#### 📖 MEDIUM PRIORITY:
```
6. Complete Vedas with translations
   → Each Veda ~50-100 MB
   → Download to: data/complete_scriptures/vedas/*.pdf

7. Dharma Shastras (Manu Smriti, etc.)
   → Each ~20-50 MB
   → Download to: data/complete_scriptures/smritis/*.pdf

8. Upanishads with commentary
   → Adi Shankaracharya commentaries
   → Download to: data/complete_scriptures/upanishads/*.pdf
```

---

## 📂 Step 2: Organize Your Files

Just drop PDFs anywhere in the appropriate category folder:

```
data/complete_scriptures/
├── puranas/
│   ├── bhagavata_purana.pdf        ← Drop here
│   ├── vishnu_purana.pdf
│   ├── shiva_purana.pdf
│   └── ... (any PDF)
│
├── itihasas/
│   ├── mahabharata.pdf             ← Drop here
│   ├── ramayana.pdf
│   └── ... (any PDF)
│
├── vedas/
│   ├── rig_veda.pdf                ← Drop here
│   ├── yajur_veda.pdf
│   └── ... (any PDF)
│
├── upanishads/
│   ├── 108_upanishads.pdf          ← Drop here
│   └── ... (any PDF)
│
└── ... (any category)
```

**The script will find and process ALL PDFs automatically!**

---

## 🚀 Step 3: Process Everything

Just run ONE command:

```bash
cd /media/rupert/New\ Volume/Testing\ Ground\ DharmaMind/Testing\ ground/DharmaMind-chat-master/dharmallm

python data/scripts/process_complete_corpus.py
```

### What It Does:
1. ✅ **Finds all PDFs** in `data/complete_scriptures/`
2. ✅ **Extracts text** from PDFs (using PyPDF2 + pdfplumber)
3. ✅ **Processes HTML files** from `data/mass_download/sacred_texts/`
4. ✅ **Processes TXT files** (Gita, Yoga Sutras, etc.)
5. ✅ **Processes JSON files** (structured data)
6. ✅ **Cleans all text** (removes headers, page numbers, artifacts)
7. ✅ **Chunks into training samples** (512 tokens each with overlap)
8. ✅ **Creates train/val/test splits** (80/10/10)
9. ✅ **Saves to JSONL format** ready for training!

### Output:
```
data/training/complete_corpus/
├── train.jsonl      ← 80% of data (ready for training!)
├── val.jsonl        ← 10% for validation
├── test.jsonl       ← 10% for testing
└── stats.json       ← Statistics (counts by format/category)
```

---

## 📊 Expected Results:

### Current Data:
- Complete scriptures: **2.8 MB** (TXT/JSON)
- Sacred-Texts download: **7.5 MB** (HTML, growing to 1-2 GB)
- **TOTAL NOW: ~10 MB**

### After Adding PDFs:
```
+ Mahabharata PDF:          200 MB
+ Bhagavata Purana PDF:     500 MB
+ All 18 Puranas PDFs:      5 GB
+ Ramayana PDF:             500 MB
+ 108 Upanishads PDF:       200 MB
+ Vedas PDFs:               400 MB
+ Other texts:              200 MB
─────────────────────────────────
TOTAL:                      ~7 GB ✅
```

**This is REAL training data!** 7 GB / 1.5B params = good data-to-parameter ratio!

---

## 🎯 Quick Start:

### RIGHT NOW (while Sacred-Texts downloads):
1. Go to Archive.org
2. Search "mahabharata ganguli pdf"
3. Download the PDF (200 MB)
4. Save to: `data/complete_scriptures/itihasas/mahabharata.pdf`
5. Search "bhagavata purana pdf"
6. Download (500 MB)
7. Save to: `data/complete_scriptures/puranas/bhagavata_purana.pdf`

### Then run:
```bash
python data/scripts/process_complete_corpus.py
```

**BOOM! Instant GB-scale training corpus!** 🚀

---

## 💡 Pro Tips:

### 1. **Check PDF Quality:**
Not all PDFs extract well. Preview the first page:
```bash
python -c "
import PyPDF2
with open('your.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(reader.pages[0].extract_text()[:500])
"
```

### 2. **Batch Download:**
If you have many PDFs in a folder:
```bash
# Just drop them all in the appropriate category folder!
# The script processes everything automatically
```

### 3. **Monitor Progress:**
The script shows real-time progress:
```
Processing PDF: mahabharata.pdf...
  Extracted 2500 chunks from PDF
Processing PDF: bhagavata_purana.pdf...
  Extracted 1800 chunks from PDF
...
```

### 4. **Check Stats:**
After processing, check `data/training/complete_corpus/stats.json`:
```json
{
  "by_format": {
    "txt": 50,
    "json": 120,
    "pdf": 8500,    ← PDFs dominate (good!)
    "html": 2300
  },
  "total_chunks": 10970
}
```

---

## ✅ Summary:

**OLD WAY:** Create separate scripts for each format, manually process
**NEW WAY:** Drop PDFs in folders, run ONE script, done!

```bash
# That's it! ONE command processes EVERYTHING:
python data/scripts/process_complete_corpus.py
```

**Now go download those GB-scale PDFs!** 📚🔥
