# 🕉️ DharmaMind AI - Advanced Spiritual Guidance Platform

> **🚀 Production-Ready** | **🔒 Security Score: 10/10** | **📅 Last Updated:** December 12, 2025

## 🌟 Overview

DharmaMind is a comprehensive spiritual AI platform combining ancient Sanatana Dharma wisdom with cutting-edge artificial intelligence. It provides personalized spiritual guidance, emotional intelligence, and transformative life insights through a modern, secure web interface.

## 🏗️ Project Structure

| App | Port | Domain | Description |
|-----|------|--------|-------------|
| **Brand Website** | 3001 | dharmamind.com | Marketing, payments, enterprise |
| **Chat Application** | 3000 | dharmamind.ai | AI-powered spiritual guidance |
| **Community** | 3002 | dharmamind.org | Forums, discussions, events |
| **Backend API** | 8000 | api.dharmamind.com | FastAPI with auth & AI services |

## 🚀 Quick Start

### Start All Services

```bash
# Start Brand Website (Port 3001)
cd Brand_Webpage && npm run dev

# Start Chat Application (Port 3000)
cd dharmamind-chat && npm run dev

# Start Community (Port 3002)
cd DhramaMind_Community && npm run dev

# Start Backend API (Port 8000)
cd backend && python start_backend.py
```

### Using VS Code Tasks

Open Command Palette (`Ctrl+Shift+P`) → "Tasks: Run Task" and select:
- 🚀 Start Brand Website (Port 3001)
- 💬 Start Chat App (Port 3000)

## 📊 Architecture

```
DharmaMind/
├── 🌐 Brand_Webpage/          # Next.js marketing site (Port 3001)
│   ├── pages/                 # Website pages
│   ├── components/            # React components
│   └── services/              # Payment & auth services
├── 💬 dharmamind-chat/        # Next.js chat app (Port 3000)
│   ├── pages/                 # Chat interface
│   ├── components/            # Chat components
│   └── services/              # AI chat services
├── 👥 DhramaMind_Community/   # Next.js community (Port 3002)
│   ├── pages/                 # Forum, discussions
│   └── components/            # Community components
├── ⚡ backend/                 # FastAPI backend (Port 8000)
│   ├── app/                   # Main application
│   │   ├── routes/            # API endpoints
│   │   ├── services/          # Business logic
│   │   ├── security/          # Security middleware
│   │   └── chakra_modules/    # AI/LLM integration
│   └── tests/                 # Test suite
├── 🧠 dharmallm/              # Custom LLM training
└── 🔮 dharmamind_vision/      # Vision system
```

## 🔒 Security Features

- ✅ **0 npm vulnerabilities** - All packages updated
- ✅ **0 Python vulnerabilities** - All CVEs patched
- ✅ **PyJWT authentication** - Secure token handling
- ✅ **CSRF protection** - Token-based middleware
- ✅ **Rate limiting** - IP-based throttling
- ✅ **XSS prevention** - DOMPurify integration
- ✅ **SQL injection protection** - Parameterized queries
- ✅ **Password validation** - Strong requirements
- ✅ **Session management** - Token blacklisting
- ✅ **Audit logging** - Security event tracking

## 🛠️ Technology Stack

### Frontend
- **Framework:** Next.js 14.2
- **Styling:** Tailwind CSS
- **State:** React Context
- **Auth:** NextAuth.js

### Backend
- **Framework:** FastAPI
- **Auth:** PyJWT + bcrypt
- **Database:** PostgreSQL + SQLAlchemy
- **Cache:** Redis
- **AI:** OpenAI, Anthropic, Custom LLM

### DevOps
- **Container:** Docker + Docker Compose
- **Monitoring:** Prometheus + Grafana
- **CI/CD:** GitHub Actions + Dependabot

## 📦 Installation

### Prerequisites
- Node.js 18+
- Python 3.12+
- PostgreSQL 15+
- Redis 7+

### Setup

```bash
# Clone repository
git clone https://github.com/Bahuncoder/dharmamindv3.git
cd DharmaMind-chat-master

# Install frontend dependencies
cd Brand_Webpage && npm install
cd ../dharmamind-chat && npm install
cd ../DhramaMind_Community && npm install

# Setup Python environment
python -m venv dharmallm_env
source dharmallm_env/bin/activate
cd backend && pip install -r requirements.txt

# Configure environment
cp backend/.env.example backend/.env
# Edit .env with your settings
```

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/security/ -v

# Test results: 34 passed, 9 skipped (endpoint tests)
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Security Management](./docs/SECRET_MANAGEMENT.md) | AWS/Vault/GCP/Azure secrets |
| [API Documentation](./backend/README.md) | Backend API reference |
| [DharmaLLM](./dharmallm/README.md) | Custom LLM training |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📜 License

This project is built with the intention of spreading authentic spiritual wisdom. See [LICENSE](./LICENSE) for details.

---

> "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।  
> सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥"
>
> _"May all beings be happy, may all beings be healthy.  
> May all beings experience prosperity, may no one suffer."_

**🕉️ Built with Reverence • Enhanced with Intelligence • Shared with Love 🕉️**
