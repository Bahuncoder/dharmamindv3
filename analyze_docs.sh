#!/bin/bash

echo "=== DharmaMind Documentation Cleanup Analysis ==="
echo 

echo "📊 Documentation Statistics:"
echo "Total MD files: $(find . -name "*.md" -type f | wc -l)"
echo "Empty MD files: $(find . -name "*.md" -type f -size 0 | wc -l)"
echo "Large files (>20KB): $(find . -name "*.md" -type f -size +20k | wc -l)"

echo
echo "🗂️ Essential Documentation (Keep):"
echo "README.md - $(ls -lh README.md 2>/dev/null | awk '{print $5}')"
echo "CONSOLIDATION_REPORT.md - $(ls -lh CONSOLIDATION_REPORT.md 2>/dev/null | awk '{print $5}')"
echo "backend/README.md - $(ls -lh backend/README.md 2>/dev/null | awk '{print $5}')"
echo "Brand_Webpage/README.md - $(ls -lh Brand_Webpage/README.md 2>/dev/null | awk '{print $5}')"

echo
echo "📜 Analysis/Status Files (Candidates for Removal):"
find . -name "*ANALYSIS*" -name "*.md" -exec ls -lh {} \; | head -10

echo
echo "📊 Complete/Summary Files (Review for Consolidation):"
find . -name "*COMPLETE*" -name "*.md" -exec ls -lh {} \; | head -10
find . -name "*SUMMARY*" -name "*.md" -exec ls -lh {} \; | head -10

echo
echo "🔍 Empty Files (Remove):"
find . -name "*.md" -type f -size 0

echo
echo "📋 Duplicate/Similar Named Files:"
echo "Architecture files:"
find . -name "*ARCHITECTURE*" -name "*.md" | head -5
echo "Assessment files:"
find . -name "*ASSESSMENT*" -name "*.md" | head -5
echo "Security files:"
find . -name "*SECURITY*" -name "*.md" | head -5