# 📚 DharmaMind Platform Documentation

Comprehensive documentation for the DharmaMind spiritual AI platform.

## 🏗️ Architecture Overview

DharmaMind is built as a microservices architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend Apps │    │  Auth Backend   │    │   DharmaLLM     │
│                 │    │                 │    │                 │
│ • Brand Website │◄──►│ • Authentication│◄──►│ • AI Models     │
│ • Chat App      │    │ • User Mgmt     │    │ • Knowledge Base│
│ • Community     │    │ • Sessions      │    │ • Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📖 Documentation Sections

### 🔌 API Documentation

- [Backend Auth API](api/backend/) - Authentication and user management
- [DharmaLLM API](api/dharmallm/) - AI and spiritual intelligence endpoints
- [Frontend APIs](api/frontend/) - Client-side API integrations

### 🚀 Deployment Guides

- [Local Development](deployment/local/) - Setting up development environment
- [Staging Deployment](deployment/staging/) - Staging environment deployment
- [Production Deployment](deployment/production/) - Production deployment guide

### 🏛️ Architecture Documentation

- [System Architecture](architecture/diagrams/) - High-level system design
- [Architecture Decisions](architecture/decisions/) - ADRs and design choices
- [Design Patterns](architecture/patterns/) - Common patterns used

### 👨‍💻 Development Guides

- [Setup Instructions](development/setup/) - Developer onboarding
- [Coding Guidelines](development/guidelines/) - Code standards and practices
- [Contributing Guide](development/contributing/) - How to contribute

## 🎯 Quick Navigation

### For Developers

1. Start with [Development Setup](development/setup/)
2. Review [Coding Guidelines](development/guidelines/)
3. Check [API Documentation](api/) for your service

### For DevOps

1. Review [Architecture Overview](architecture/)
2. Follow [Deployment Guides](deployment/)
3. Check [Infrastructure Documentation](../infrastructure/)

### For Product Teams

1. Understand [System Architecture](architecture/diagrams/)
2. Review [API Capabilities](api/)
3. Check [Architecture Decisions](architecture/decisions/)

## 📋 Documentation Standards

### Writing Guidelines

- Use clear, concise language
- Include code examples where relevant
- Keep documentation up-to-date with code changes
- Use consistent formatting and structure

### Diagram Standards

- Use PlantUML for architecture diagrams
- Include both high-level and detailed views
- Update diagrams when architecture changes
- Store diagrams as code for version control

## 🔄 Keeping Documentation Updated

This documentation should be updated whenever:

- New APIs are added or changed
- Architecture decisions are made
- Deployment processes change
- New development patterns are established

## 🆘 Getting Help

- **Development Questions**: Check development guides or ask in dev channels
- **Deployment Issues**: Refer to deployment guides or contact DevOps
- **Architecture Questions**: Review architecture docs or reach out to architects

---

## 📝 Contributing to Documentation

1. Follow the established structure
2. Use clear headings and sections
3. Include practical examples
4. Update related documents when making changes
5. Review documentation in pull requests

For detailed contributing guidelines, see [Contributing Guide](development/contributing/).
