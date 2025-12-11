# Contributing to DharmaMind

Thank you for your interest in contributing to DharmaMind! This document provides guidelines and information for contributors.

## 🌟 Code of Conduct

This project adheres to principles of compassion, wisdom, and mindful development. We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Values
- **Compassion**: Treat all contributors with kindness and understanding
- **Wisdom**: Make thoughtful decisions that benefit the community
- **Mindfulness**: Be present and conscious in all interactions
- **Non-harm**: Ensure contributions do not cause harm to users or community
- **Right Intention**: Contribute with genuine desire to help and improve

## 🚀 Getting Started

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/dharmamind.git
   cd dharmamind
   ```

2. **Set Up Development Environment**
   ```bash
   # Backend setup
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Frontend setup
   cd ../frontend
   npm install
   ```

3. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your development configuration
   ```

4. **Run Tests**
   ```bash
   # Backend tests
   cd backend
   python -m pytest

   # Frontend tests
   cd frontend
   npm test
   ```

## 📝 How to Contribute

### Types of Contributions

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve docs, guides, and examples
5. **Testing**: Add or improve test coverage
6. **Spiritual Content**: Contribute dharmic wisdom and guidance

### Reporting Issues

When reporting bugs, please include:

- **Clear Description**: What happened vs. what you expected
- **Steps to Reproduce**: Detailed steps to recreate the issue
- **Environment**: OS, Python version, Node.js version, etc.
- **Screenshots**: If applicable, include screenshots
- **Logs**: Relevant error messages or logs

Use our issue templates:
- Bug Report
- Feature Request
- Security Issue
- Documentation Improvement

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Changes**
   - Follow our coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new authentication feature"
   # Follow conventional commit format
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then create a Pull Request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/GIFs if applicable
   - Checklist completion

## 🎯 Coding Standards

### Python (Backend)

1. **Code Style**
   ```bash
   # Use Black for formatting
   black backend/

   # Use isort for imports
   isort backend/

   # Use flake8 for linting
   flake8 backend/
   ```

2. **Standards**
   - Follow PEP 8
   - Use type hints
   - Write docstrings for functions/classes
   - Maximum line length: 88 characters

3. **Example**
   ```python
   from typing import Optional
   
   def authenticate_user(email: str, password: str) -> Optional[User]:
       """
       Authenticate user with email and password.
       
       Args:
           email: User's email address
           password: User's password
           
       Returns:
           User object if authentication successful, None otherwise
       """
       # Implementation here
       pass
   ```

### TypeScript (Frontend)

1. **Code Style**
   ```bash
   # Use Prettier for formatting
   npm run format

   # Use ESLint for linting
   npm run lint
   ```

2. **Standards**
   - Use TypeScript strictly
   - Follow React best practices
   - Use functional components with hooks
   - Implement proper error handling

3. **Example**
   ```typescript
   interface AuthResponse {
     user: User;
     token: string;
   }
   
   const authenticateUser = async (
     email: string, 
     password: string
   ): Promise<AuthResponse> => {
     // Implementation here
   };
   ```

### Documentation

1. **Code Documentation**
   - Document all public APIs
   - Include usage examples
   - Explain complex algorithms
   - Use clear, concise language

2. **README Updates**
   - Update installation instructions
   - Add new feature documentation
   - Include configuration examples
   - Update troubleshooting guides

## 🧪 Testing Guidelines

### Backend Testing

1. **Test Structure**
   ```python
   # tests/test_auth.py
   import pytest
   from app.auth import authenticate_user
   
   def test_successful_authentication():
       """Test successful user authentication."""
       # Test implementation
       
   def test_failed_authentication():
       """Test failed authentication scenarios."""
       # Test implementation
   ```

2. **Coverage Requirements**
   - Aim for 80%+ test coverage
   - Test happy paths and edge cases
   - Include integration tests
   - Mock external dependencies

### Frontend Testing

1. **Component Testing**
   ```typescript
   // components/__tests__/LoginForm.test.tsx
   import { render, screen, fireEvent } from '@testing-library/react';
   import LoginForm from '../LoginForm';
   
   test('submits form with valid credentials', () => {
     // Test implementation
   });
   ```

2. **Testing Standards**
   - Test user interactions
   - Mock API calls
   - Test error scenarios
   - Include accessibility tests

## 🔒 Security Guidelines

### Security Best Practices

1. **Never commit secrets**
   - Use environment variables
   - Add sensitive files to .gitignore
   - Use proper secret management

2. **Input Validation**
   - Validate all user inputs
   - Sanitize data before processing
   - Use parameterized queries

3. **Authentication**
   - Follow OAuth standards
   - Use secure session management
   - Implement proper rate limiting

### Reporting Security Issues

For security vulnerabilities:
1. **DO NOT** create public issues
2. Email: security@dharmamind.ai
3. Include detailed reproduction steps
4. Allow time for responsible disclosure

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventional format
- [ ] No secrets or sensitive data committed
- [ ] PR description is clear and complete
- [ ] Related issues are referenced

## 🎨 Spiritual Content Guidelines

When contributing spiritual content:

1. **Accuracy**: Ensure authenticity to source traditions
2. **Inclusivity**: Respect all spiritual paths
3. **Non-dogmatic**: Avoid sectarian or exclusive language
4. **Practical**: Include actionable guidance
5. **Compassionate**: Use kind, understanding language

### Content Review Process

Spiritual content will be reviewed by:
- Technical maintainers
- Spiritual content advisors
- Community feedback

## 🤝 Community

### Communication Channels

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Email**: contact@dharmamind.ai for direct communication

### Community Guidelines

1. **Be Respectful**: Treat all community members with kindness
2. **Be Patient**: Remember that people have different backgrounds
3. **Be Constructive**: Provide helpful feedback and suggestions
4. **Be Mindful**: Consider the impact of your words and actions

## 🏆 Recognition

Contributors will be recognized through:

1. **Contributors File**: Listed in CONTRIBUTORS.md
2. **Release Notes**: Credited in release announcements
3. **Hall of Fame**: Featured on project website
4. **Dharma Points**: Community recognition system (future)

## 📚 Resources

### Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Docker Documentation](https://docs.docker.com/)
- [Buddhist Programming Principles](https://github.com/dharmamind/resources)

### Development Tools

- **IDE**: VS Code with recommended extensions
- **Database**: PostgreSQL with pgAdmin
- **API Testing**: Postman or Insomnia
- **Git GUI**: GitKraken or SourceTree

## ❓ Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search GitHub issues/discussions
3. Create a new discussion
4. Email: contributors@dharmamind.ai

---

**Thank you for contributing to DharmaMind! Together, we're building technology that serves wisdom and compassion.** 🙏
