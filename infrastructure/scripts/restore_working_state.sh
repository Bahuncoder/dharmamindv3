#!/bin/bash

# DharmaMind Working State Restoration Script
# This script restores the system to the known working state with Brand_Webpage functional

echo "🔄 Restoring DharmaMind to working state with Brand_Webpage functional..."

# Store current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Stash any uncommitted changes
echo "💾 Stashing any uncommitted changes..."
git stash push -m "Auto-stash before restore - $(date)"

# Restore to the working commit
echo "🎯 Restoring to working commit (7551da5)..."
git checkout 7551da5

# Alternative: Restore from tag
echo "🏷️  Or restore from tag..."
# git checkout v1.0-brand-webpage-working

# Alternative: Restore from backup branch
echo "🔄 Or restore from backup branch..."
# git checkout backup-working-brand-webpage

echo ""
echo "✅ RESTORATION OPTIONS:"
echo "1. Current: Reset to commit 7551da5"
echo "2. Tag:     git checkout v1.0-brand-webpage-working"
echo "3. Branch:  git checkout backup-working-brand-webpage"
echo ""
echo "🚀 To start Brand_Webpage:"
echo "   cd Brand_Webpage && npm run dev"
echo ""
echo "📦 Dependencies should be installed. If not:"
echo "   cd Brand_Webpage && npm install"
echo ""
echo "🌐 Brand_Webpage will be available at: http://localhost:3001"
echo ""
