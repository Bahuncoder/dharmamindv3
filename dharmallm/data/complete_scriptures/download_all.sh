#!/bin/bash
# Download Complete Hindu Scriptures from Archive.org
# Run this script to download all texts

echo "🕉️ Downloading Complete Hindu Scriptures..."
echo "=============================================="

# Create directories
mkdir -p data/complete_scriptures/{vedas,puranas,smritis,itihasas,gita}

# Download Vedas (if URLs work)
echo "📖 Downloading Vedas..."
wget -O data/complete_scriptures/vedas/rig_veda.txt https://archive.org/download/rigvedaenglish/rigveda.txt 2>/dev/null || echo "  ⚠️ Rig Veda URL not found - search Archive.org manually"
wget -O data/complete_scriptures/vedas/sama_veda.txt https://archive.org/download/sama-veda/SamaVeda.txt 2>/dev/null || echo "  ⚠️ Sama Veda URL not found"
wget -O data/complete_scriptures/vedas/yajur_veda.txt https://archive.org/download/yajurveda/YajurVeda.txt 2>/dev/null || echo "  ⚠️ Yajur Veda URL not found"
wget -O data/complete_scriptures/vedas/atharva_veda.txt https://archive.org/download/atharva-veda-samhita/AtharvaVeda.txt 2>/dev/null || echo "  ⚠️ Atharva Veda URL not found"

# Download Puranas
echo "📖 Downloading Puranas..."
wget -O data/complete_scriptures/puranas/vishnu_purana.txt https://archive.org/download/vishnu-purana/VishnuPurana.txt 2>/dev/null || echo "  ⚠️ Vishnu Purana URL not found"
wget -O data/complete_scriptures/puranas/bhagavata_purana.txt https://archive.org/download/SrimadBhagavatamEnglish/SrimadBhagavatam.txt 2>/dev/null || echo "  ⚠️ Bhagavata Purana URL not found"

# Download Smritis
echo "📖 Downloading Smritis..."
wget -O data/complete_scriptures/smritis/manu_smriti.txt https://archive.org/download/manu-smriti/ManuSmriti.txt 2>/dev/null || echo "  ⚠️ Manu Smriti URL not found"

# Download Itihasas
echo "📖 Downloading Itihasas..."
wget -O data/complete_scriptures/itihasas/mahabharata.txt https://archive.org/download/mahabharata-complete/Mahabharata.txt 2>/dev/null || echo "  ⚠️ Mahabharata URL not found"
wget -O data/complete_scriptures/itihasas/ramayana.txt https://archive.org/download/valmiki-ramayana/Ramayana.txt 2>/dev/null || echo "  ⚠️ Ramayana URL not found"

# Download Bhagavad Gita
echo "📖 Downloading Bhagavad Gita..."
wget -O data/complete_scriptures/gita/bhagavad_gita.txt https://archive.org/download/bhagavad-gita-all/BhagavadGita.txt 2>/dev/null || echo "  ⚠️ Bhagavad Gita URL not found"

echo ""
echo "=============================================="
echo "✅ Download attempt complete!"
echo "⚠️ Note: Many URLs may not work - Archive.org changes frequently"
echo ""
echo "🎯 MANUAL DOWNLOAD RECOMMENDED:"
echo "1. Visit https://archive.org"
echo "2. Search for each text by name"
echo "3. Download TXT or PDF format"
echo "4. Save to appropriate directory"
echo ""
echo "📁 See MANUAL_DOWNLOAD_GUIDE.md for detailed instructions"
