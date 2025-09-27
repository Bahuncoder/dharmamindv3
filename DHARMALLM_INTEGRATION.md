# 🕉️ DharmaLLM Integration - Complete Microservice Architecture

## Overview

DharmaLLM has been fully integrated as a **separate microservice**, ensuring clean architecture separation between the authentication backend and AI processing capabilities.

## Architecture

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Backend       │◄──────────────►│   DharmaLLM     │
│   (Port 8000)   │                │   (Port 8001)   │
│                 │                │                 │
│ • Authentication│                │ • AI Processing │
│ • API Gateway   │                │ • Spiritual AI  │
│ • User Mgmt     │                │ • Chat Engine   │
│ • HTTP Client   │                │ • FastAPI       │
└─────────────────┘                └─────────────────┘
        │                                   │
        │                                   │
        ▼                                   ▼
┌─────────────────┐                ┌─────────────────┐
│   Database      │                │   AI Models     │
│   (PostgreSQL)  │                │   & Cache       │
└─────────────────┘                └─────────────────┘
```

## Key Components

### 1. Backend Service (`backend/`)

- **Purpose**: Clean authentication and API gateway service
- **Port**: 8000
- **Responsibilities**:
  - User authentication and authorization
  - Session management
  - API routing and gateway functionality
  - HTTP client to communicate with DharmaLLM
  - Database operations

### 2. DharmaLLM Service (`dharmallm/`)

- **Purpose**: Independent spiritual AI processing microservice
- **Port**: 8001
- **Responsibilities**:
  - Spiritual AI chat processing
  - Dharmic wisdom generation
  - Model loading and inference
  - FastAPI service with health checks

### 3. Integration Layer

- **Communication**: HTTP REST API
- **Client**: `backend/app/services/dharma_llm_service.py`
- **Service**: `dharmallm/api/main.py`
- **Endpoints**:
  - `POST /api/v1/chat` - Main chat processing
  - `POST /api/v1/wisdom` - Spiritual wisdom queries
  - `GET /health` - Health check
  - `GET /status` - System status

## File Structure

```
project_root/
├── backend/                          # Clean backend service
│   ├── app/
│   │   ├── services/
│   │   │   └── dharma_llm_service.py # HTTP client to DharmaLLM
│   │   ├── routes/
│   │   │   ├── chat.py               # Basic chat routes
│   │   │   └── dharmic_chat.py       # Dharmic chat routes
│   │   └── config.py                 # DHARMALLM_SERVICE_URL config
│   └── Dockerfile
├── dharmallm/                        # Independent AI service
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI application
│   ├── requirements.txt              # AI service dependencies
│   └── Dockerfile                    # AI service container
└── docker-compose.yml               # Multi-service orchestration
```

## Environment Configuration

### Backend Environment Variables

```bash
# DharmaLLM Service Integration
DHARMALLM_SERVICE_URL=http://dharmallm:8001
DHARMALLM_SERVICE_TIMEOUT=30
```

### Docker Compose Services

```yaml
services:
  backend:
    # Clean authentication backend
    depends_on:
      - dharmallm # Waits for AI service
    environment:
      DHARMALLM_SERVICE_URL: http://dharmallm:8001

  dharmallm:
    # Independent AI processing service
    build: ./dharmallm
    ports:
      - "8001:8001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
```

## API Integration

### Backend → DharmaLLM Communication

```python
# backend/app/services/dharma_llm_service.py
class DharmaLLMClient:
    async def process_chat(self, message: str, session_id: str):
        # HTTP call to DharmaLLM service
        payload = {"message": message, "session_id": session_id}
        response = await self.client.post("/api/v1/chat", json=payload)
        return response.json()
```

### DharmaLLM Service Endpoints

```python
# dharmallm/api/main.py
@app.post("/api/v1/chat")
async def process_chat(request: ChatRequest):
    # AI processing logic
    result = await process_dharmic_query(request.message)
    return ChatResponse(...)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}
```

## Deployment

### Local Development

```bash
# Start individual services
cd dharmallm && python -m dharmallm.api.main  # Port 8001
cd backend && python run_server.py            # Port 8000
```

### Docker Compose (Recommended)

```bash
# Build all services
docker-compose build

# Start the complete stack
docker-compose up -d

# Check service health
docker-compose ps
curl http://localhost:8001/health  # DharmaLLM
curl http://localhost:8000/health  # Backend
```

### Service Dependencies

1. **Database** (PostgreSQL, Redis) - Started first
2. **DharmaLLM** - AI service starts independently
3. **Backend** - Waits for DharmaLLM to be healthy
4. **Frontend** - Connects to backend

## Benefits of This Architecture

### 1. **Clean Separation of Concerns**

- Backend focuses solely on authentication and API gateway
- DharmaLLM focuses solely on AI processing
- No mixed dependencies or responsibilities

### 2. **Independent Scalability**

- Scale backend for user load
- Scale DharmaLLM for AI processing load
- Different resource requirements per service

### 3. **Technology Independence**

- Backend can evolve without affecting AI models
- DharmaLLM can update models without backend changes
- Clear API contract between services

### 4. **Development Efficiency**

- Teams can work on services independently
- Faster development cycles
- Easier testing and debugging

### 5. **Production Readiness**

- Each service can be deployed to different servers
- Independent monitoring and logging
- Failure isolation between services

## Testing the Integration

```bash
# Run integration test script
./test_dharmallm_integration.sh

# Manual testing
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is dharma?", "session_id": "test"}'

# Backend proxy test
curl -X POST http://localhost:8000/api/v1/dharmic/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about meditation"}'
```

## Future Enhancements

1. **Load Balancing**: Multiple DharmaLLM instances
2. **Caching**: Redis caching layer for AI responses
3. **Monitoring**: Prometheus metrics for both services
4. **Authentication**: JWT token passing between services
5. **Model Management**: Dynamic model loading and switching

---

## Status: ✅ FULLY INTEGRATED

- [x] Clean backend architecture (authentication only)
- [x] Independent DharmaLLM microservice
- [x] HTTP-based communication
- [x] Docker Compose orchestration
- [x] Health checks and monitoring
- [x] Production-ready deployment configuration

**The DharmaLLM integration is complete and ready for production deployment.**

🙏 May this architecture serve all beings with wisdom and efficiency.
