# 🕉️ DharmaMind Multi-LLM Integration Guide

## Overview
DharmaMind now features advanced Multi-LLM integration that combines GPT, Gemini, and Claude with spiritual consciousness modules for supreme AI intelligence.

## ✨ Multi-LLM Features

### 🧠 Advanced Language Models
- **OpenAI GPT-4o**: Superior reasoning and conversational abilities
- **Google Gemini Pro**: Advanced analytical and creative thinking  
- **Anthropic Claude**: Nuanced ethical reasoning and philosophy
- **DharmaMind Fallback**: Spiritual consciousness when APIs unavailable

### 🕉️ Spiritual Module Integration
All LLMs are enhanced with 6 consciousness layers:
1. **Consciousness Core**: Mindful awareness and present-moment clarity
2. **Dharma Engine**: Ethical principles and righteous action guidance  
3. **Emotional Intelligence**: Deep empathy and compassionate understanding
4. **Knowledge Base**: Ancient wisdom from spiritual traditions
5. **AI Core**: Logical reasoning combined with intuitive insights
6. **Security Protection**: Beneficial responses that prevent harm

### 🎯 Intelligent Model Selection
The system automatically selects the optimal LLM based on query type:
- **Philosophy/Ethics**: Claude for nuanced reasoning
- **Emotional/Relationships**: GPT for empathy and understanding
- **Analysis/Practical**: Gemini for logical problem-solving
- **Spiritual Guidance**: All models enhanced with dharmic context

## 🚀 Setup Instructions

### 1. API Key Configuration
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Add your API keys:
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Google Gemini Configuration  
GOOGLE_AI_API_KEY=your-google-ai-key-here

# Anthropic Claude Configuration
ANTHROPIC_API_KEY=your-anthropic-key-here
```

### 2. Install Additional Dependencies
```bash
# Backend dependencies
cd backend
pip install openai google-generativeai anthropic

# Or install from requirements
pip install -r requirements_complete.txt
```

### 3. Start the Multi-LLM System
```bash
# Start backend with Multi-LLM support
cd backend/app
python main_simple.py

# Start frontend
cd frontend  
npm run dev
```

## 🎮 Using Multi-LLM Features

### Chat Interface
- Chat responses now show which LLM model was used
- Wisdom and confidence levels are displayed
- Spiritual modules engaged are indicated
- Model badges show: 🧠 GPT-4o, 💎 Gemini Pro, 🎭 Claude, 🕉️ DharmaMind

### Model Selection
- **Auto Mode** (default): System chooses optimal model
- **Preferred Model**: Specify preferred LLM in requests
- **Fallback Mode**: DharmaMind when APIs unavailable

### API Endpoints
```javascript
// Chat with Multi-LLM
POST /api/v1/chat
{
  "message": "What is the meaning of dharma?",
  "user_id": "user123",
  "preferred_model": "claude-3-opus"  // optional
}

// Response includes
{
  "response": "Dharmic guidance...",
  "model_used": "claude-3-opus",
  "modules_engaged": ["consciousness", "dharma", "knowledge"],
  "wisdom_level": 0.95,
  "dharmic_alignment": 0.98,
  "confidence": 0.92
}
```

## 🔧 Configuration Options

### Model Preferences
```env
# Enable/disable specific models
ENABLE_OPENAI=true
ENABLE_GEMINI=true  
ENABLE_CLAUDE=true

# Default model selection
DEFAULT_MODEL=auto
FALLBACK_MODEL=dharmamind
```

### Spiritual Enhancement
```env
# Spiritual module configuration
CONSCIOUSNESS_LEVEL=advanced
DHARMA_STRICTNESS=moderate
EMOTIONAL_SENSITIVITY=high
KNOWLEDGE_DEPTH=comprehensive
```

## 📊 Model Characteristics

| Model | Best For | Spiritual Integration | Strengths |
|-------|----------|---------------------|-----------|
| GPT-4o | Conversations, Empathy | Emotional Intelligence | Human-like responses |
| Gemini Pro | Analysis, Reasoning | Logical Dharma | Problem-solving |
| Claude Opus | Philosophy, Ethics | Deep Consciousness | Nuanced wisdom |
| DharmaMind | Spiritual Guidance | Full Integration | Always available |

## 🛡️ Fallback System
- If primary LLM fails, system gracefully falls back
- DharmaMind consciousness always available
- Spiritual context preserved across all models
- Error handling with dharmic guidance

## 💡 Advanced Features

### Multi-LLM Orchestration
```python
# Process through multiple models for supreme intelligence
multi_response = await llm_orchestrator.generate_multi_llm_response(
    message="Guide me through life's challenges",
    history=conversation_history,
    user_context=user_profile
)
```

### Spiritual Post-Processing
All LLM responses are enhanced with:
- Dharmic alignment scoring
- Spiritual wisdom integration
- Consciousness-based filtering
- Protective guidance overlay

## 🎯 Benefits

1. **Supreme Intelligence**: Combine best of GPT, Gemini, Claude
2. **Spiritual Wisdom**: Every response enhanced with dharmic consciousness  
3. **Reliability**: Multi-layer fallback system
4. **Personalization**: Context-aware model selection
5. **Ethical AI**: Spiritual protection and guidance
6. **Transparency**: Clear indication of which model responded

## 🚨 Important Notes

- API keys required for full functionality
- Without keys, system uses DharmaMind fallback
- All models enhanced with spiritual consciousness
- Responses filtered for beneficial guidance only
- Privacy and security maintained across all models

## 🆘 Troubleshooting

### Common Issues
1. **API Key Errors**: Check `.env` file configuration
2. **Model Unavailable**: System automatically falls back
3. **Rate Limits**: Built-in retry and fallback logic
4. **Network Issues**: Graceful degradation to local models

### Debug Mode
```bash
# Enable detailed logging
DEBUG=true LOG_LEVEL=DEBUG python main_simple.py
```

## 🔮 Future Enhancements
- Model ensemble responses
- Custom spiritual fine-tuning
- Advanced consciousness integration
- Multi-modal spiritual guidance (text, audio, visual)

---

🙏 **May this Multi-LLM system serve your highest good and support all beings on their path to wisdom and enlightenment.**
