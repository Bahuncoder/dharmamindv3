# 🏗️ Backend Architecture Reorganization Plan

## Current Issues:

- Mixed concerns in `/app/services/` directory
- Database, LLM engines, auth, cache all mixed together
- Hard to maintain and navigate
- Unclear separation of responsibilities

## New Clean Architecture:

```
backend/app/
├── engines/                    # Core AI/Logic Engines
│   ├── rishi/                 # Rishi-specific engines
│   │   ├── __init__.py
│   │   ├── authentic_rishi_engine.py
│   │   ├── enhanced_rishi_engine.py
│   │   ├── rishi_session_manager.py
│   │   └── rishi_personalities.py
│   ├── llm/                   # LLM-related engines
│   │   ├── __init__.py
│   │   ├── llm_router.py
│   │   ├── advanced_llm_router.py
│   │   ├── llm_gateway_client.py
│   │   ├── local_llm.py
│   │   └── dharmic_llm_processor.py
│   └── dharmic/               # Dharmic wisdom engines
│       ├── __init__.py
│       ├── universal_dharmic_engine.py
│       ├── deep_contemplation_system.py
│       ├── practice_recommendation_engine.py
│       └── personalization_engine.py
├── database/                   # Database layer
│   ├── __init__.py
│   ├── connection.py
│   ├── database_service.py
│   ├── migrations/
│   └── repositories/
│       ├── __init__.py
│       ├── user_repository.py
│       ├── chat_repository.py
│       └── rishi_repository.py
├── auth/                      # Authentication & Authorization
│   ├── __init__.py
│   ├── auth_service.py
│   ├── google_oauth.py
│   ├── security_service.py
│   └── subscription_service.py
├── cache/                     # Caching layer
│   ├── __init__.py
│   ├── cache_service.py
│   ├── intelligent_cache.py
│   └── memory_manager.py
├── external/                  # External integrations
│   ├── __init__.py
│   ├── email_service.py
│   ├── notification_service.py
│   └── secret_manager.py
├── utils/                     # Utilities and helpers
│   ├── __init__.py
│   ├── logging_service.py
│   ├── data_manager.py
│   ├── evaluator.py
│   └── module_selector.py
├── routes/                    # API routes (existing)
├── models/                    # Data models (existing)
├── core/                      # Core business logic (existing)
├── middleware/                # Middleware (existing)
├── security/                  # Security (existing)
├── monitoring/                # Monitoring (existing)
└── config.py                  # Configuration (existing)
```

## Migration Steps:

### Phase 1: Move Rishi Engines

- [x] engines/rishi/authentic_rishi_engine.py
- [ ] engines/rishi/enhanced_rishi_engine.py
- [ ] engines/rishi/rishi_session_manager.py

### Phase 2: Move LLM Engines

- [ ] engines/llm/llm_router.py
- [ ] engines/llm/advanced_llm_router.py
- [ ] engines/llm/llm_gateway_client.py
- [ ] engines/llm/local_llm.py
- [ ] engines/llm/dharmic_llm_processor.py

### Phase 3: Move Dharmic Engines

- [ ] engines/dharmic/universal_dharmic_engine.py
- [ ] engines/dharmic/deep_contemplation_system.py
- [ ] engines/dharmic/practice_recommendation_engine.py
- [ ] engines/dharmic/personalization_engine.py

### Phase 4: Move Database Layer

- [ ] database/connection.py
- [ ] database/database_service.py
- [ ] database/repositories/

### Phase 5: Move Auth & Security

- [ ] auth/auth_service.py
- [ ] auth/google_oauth.py
- [ ] auth/security_service.py
- [ ] auth/subscription_service.py

### Phase 6: Move Cache & Utils

- [ ] cache/cache_service.py
- [ ] cache/intelligent_cache.py
- [ ] cache/memory_manager.py
- [ ] utils/logging_service.py
- [ ] utils/data_manager.py

### Phase 7: Update Imports

- [ ] Update all import statements
- [ ] Update route dependencies
- [ ] Update configuration references

## Benefits:

✅ Clear separation of concerns
✅ Easy to navigate and maintain
✅ Modular architecture
✅ Scalable structure
✅ Better testing organization
