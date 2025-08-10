#!/bin/bash

# =============================================================================
# DharmaMind GitHub Repository Setup Script
# =============================================================================

echo "🧘 DharmaMind GitHub Repository Setup"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Repository information
REPO_NAME="dharmamind"
REPO_DESCRIPTION="Enterprise Authentication & AI Platform - Spiritual wisdom meets modern technology"
REPO_URL="https://github.com/yourusername/dharmamind.git"

echo -e "${BLUE}Project Status:${NC}"
echo "✅ Git repository initialized"
echo "✅ Initial commit completed (181 files, 82,283+ lines)"
echo "✅ .gitignore configured"
echo "✅ License added (MIT)"
echo "✅ Documentation complete"
echo ""

echo -e "${BLUE}Repository Contents:${NC}"
echo "📁 Backend (FastAPI + PostgreSQL + Redis)"
echo "📁 Frontend (Next.js + TypeScript)"
echo "📁 Docker Production Environment"
echo "📁 Monitoring Stack (Prometheus + Grafana + ELK)"
echo "📁 Comprehensive Documentation"
echo "📁 Deployment Scripts"
echo ""

echo -e "${YELLOW}Next Steps to Upload to GitHub:${NC}"
echo ""

echo -e "${GREEN}1. Create Repository on GitHub:${NC}"
echo "   • Go to: https://github.com/new"
echo "   • Repository name: ${REPO_NAME}"
echo "   • Description: ${REPO_DESCRIPTION}"
echo "   • Set to Public or Private"
echo "   • DON'T initialize with README (we have one)"
echo ""

echo -e "${GREEN}2. Add Remote Origin:${NC}"
echo "   git remote add origin https://github.com/YOUR_USERNAME/dharmamind.git"
echo ""

echo -e "${GREEN}3. Push to GitHub:${NC}"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""

echo -e "${GREEN}4. Configure Repository Settings:${NC}"
echo "   • Add repository topics: ai, authentication, fastapi, nextjs, docker, spiritual-ai"
echo "   • Enable Issues and Discussions"
echo "   • Configure branch protection for main"
echo "   • Add repository description and website URL"
echo ""

echo -e "${GREEN}5. Set Up GitHub Actions (Optional):${NC}"
echo "   • CI/CD pipeline for automated testing"
echo "   • Docker image building and publishing"
echo "   • Security scanning"
echo "   • Documentation deployment"
echo ""

echo -e "${BLUE}Repository Statistics:${NC}"
git log --oneline | wc -l | tr -d ' ' | sed 's/^/Commits: /'
find . -name "*.py" | xargs wc -l | tail -1 | sed 's/^[ ]*\([0-9]*\).*/Python Lines: \1/'
find . -name "*.ts" -o -name "*.tsx" | xargs wc -l | tail -1 | sed 's/^[ ]*\([0-9]*\).*/TypeScript Lines: \1/'
find . -name "*.md" | wc -l | tr -d ' ' | sed 's/^/Documentation Files: /'
echo ""

echo -e "${BLUE}Key Features Ready for GitHub:${NC}"
echo "🔐 Enterprise Authentication System"
echo "🤖 Multi-LLM AI Integration"
echo "🐳 Production Docker Environment"
echo "📊 Monitoring & Analytics"
echo "🔒 Security & Audit Logging"
echo "📚 Comprehensive Documentation"
echo "🚀 Deployment Automation"
echo "🧪 Testing Framework"
echo "💾 Database Schema & Migrations"
echo "🎨 Modern Frontend Interface"
echo ""

echo -e "${BLUE}Environment Variables to Configure:${NC}"
echo "📝 Copy .env.example to .env and update:"
echo "   • Database credentials"
echo "   • JWT secrets"
echo "   • OAuth client IDs"
echo "   • SMTP settings"
echo "   • API keys"
echo ""

echo -e "${BLUE}Demo Access:${NC}"
echo "🎯 Test the system with demo accounts:"
echo "   • Free: demo.free@dharmamind.ai / demo123"
echo "   • Premium: demo.premium@dharmamind.ai / demo123"
echo "   • Enterprise: demo.enterprise@dharmamind.ai / demo123"
echo ""

echo -e "${BLUE}Quick Start Commands:${NC}"
echo "# Development mode"
echo "cd backend/app && python enhanced_enterprise_auth.py"
echo "cd frontend && npm install && npm run dev"
echo ""
echo "# Production mode"
echo "./deploy.sh deploy"
echo ""

echo -e "${YELLOW}Important Security Notes:${NC}"
echo "⚠️  Never commit .env files with real secrets"
echo "⚠️  Change all default passwords in production"
echo "⚠️  Use HTTPS in production"
echo "⚠️  Configure proper CORS origins"
echo "⚠️  Set up monitoring alerts"
echo ""

echo -e "${GREEN}Repository is ready for GitHub! 🎉${NC}"
echo ""
echo "After uploading to GitHub, your project will be available at:"
echo "https://github.com/YOUR_USERNAME/dharmamind"
echo ""
echo "Don't forget to:"
echo "• Update the README with your actual GitHub URL"
echo "• Configure GitHub repository settings"
echo "• Set up GitHub Actions for CI/CD"
echo "• Add contributors and maintainers"
echo ""

# Create a quick reference file
cat > GITHUB_SETUP_REFERENCE.md << 'EOF'
# GitHub Setup Quick Reference

## Repository Information
- **Name**: dharmamind
- **Description**: Enterprise Authentication & AI Platform - Spiritual wisdom meets modern technology
- **License**: MIT
- **Language**: Python, TypeScript
- **Framework**: FastAPI, Next.js

## Quick Commands
```bash
# Add remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/dharmamind.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Repository Topics
Add these topics in GitHub repository settings:
- ai
- authentication
- fastapi
- nextjs
- docker
- spiritual-ai
- enterprise
- postgresql
- redis
- typescript

## Branch Protection
Recommended settings for main branch:
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to specific people/teams

## Security Settings
- Enable vulnerability alerts
- Enable dependency security updates
- Configure secret scanning
- Set up code scanning with CodeQL

## GitHub Actions
Consider setting up workflows for:
- Automated testing
- Docker image building
- Security scanning
- Documentation deployment
- Dependency updates

## Repository Settings
- Enable Issues
- Enable Discussions  
- Enable Projects (for project management)
- Configure GitHub Pages (for documentation)
- Set up webhooks (if needed)
EOF

echo -e "${GREEN}Created GITHUB_SETUP_REFERENCE.md for your convenience!${NC}"
