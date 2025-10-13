# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calma Backend is a NestJS-based REST API that provides culturally-aware mental health chat services. It integrates with a FastAPI AI service (`calma-ai/`) for psychological support with cultural context awareness and conversation memory management.

## Architecture

### Core Technology Stack
- **Framework**: NestJS with TypeScript
- **Database**: PostgreSQL with Prisma ORM
- **Authentication**: JWT with Passport
- **External Services**: FastAPI AI service (localhost:8000)
- **CORS**: Configured for frontend on ports 8080, 5173, 3001

### Database Architecture
The Prisma schema (`src/database/prisma/schema.prisma`) implements a sophisticated mental health platform with:

**Core Models:**
- `User`: Authentication and profile management with role-based access
- `Chat`: Enhanced chat messages with AI context (emotional tone, detected topics, cultural adaptations)
- `CulturalProfile`: Comprehensive cultural demographics (age, location, education, ethnicity, religion, family structure, economic status)
- `ConversationSession`: Memory-grouped conversations with AI-generated insights and action items
- `UserTopic`: Cross-session topic tracking with severity and status monitoring
- `ProgressTracking`: Relationship strength, trust levels, and improvement tracking over time

**Key Features:**
- Cultural awareness with Zimbabwe-focused enums (Shona/Ndebele languages, urban/rural contexts)
- Conversation memory system with session grouping and topic persistence
- AI-enhanced chat with emotional tone detection and cultural adaptations
- Progress tracking with relationship building metrics

### Module Structure

**Authentication & Users:**
- `src/auth/`: JWT authentication with cookie-based sessions
- `src/user/`: User management and profiles

**Core Chat Services:**
- `src/chat/`: Main chatbot interaction with AI service integration
- `src/therapist-chat/`: Human therapist communication
- `src/conversation-session/`: Session management and memory persistence
- `src/cultural-profile/`: Cultural context management

**Support Features:**
- `src/mood/`: Mood tracking with cultural factors
- `src/resources/`: Mental health resources with cultural relevance
- `src/saved-resource/`: User's saved resources

**Infrastructure:**
- `src/database/`: Prisma service and database module
- `src/common/`: Shared interfaces for AI service communication

## Development Commands

### Environment Setup
```bash
# Install dependencies
npm install

# Set up Prisma database
npx prisma generate
npx prisma db push  # For development
npx prisma migrate dev  # For production migrations
```

### Development
```bash
# Development with hot reload
npm run start:dev

# Debug mode
npm run start:debug

# Production build and start
npm run build
npm run start:prod
```

### Code Quality
```bash
# Linting (auto-fix)
npm run lint

# Code formatting
npm run format

# Type checking (via build process)
npm run build
```

### Testing
```bash
# Unit tests
npm run test

# Watch mode for development
npm run test:watch

# End-to-end tests
npm run test:e2e

# Test coverage
npm run test:cov

# Debug tests
npm run test:debug
```

### Database Operations
```bash
# Generate Prisma client after schema changes
npx prisma generate

# Apply schema changes to database
npx prisma db push

# Create and apply migrations
npx prisma migrate dev

# Reset database (development only)
npx prisma migrate reset

# Open Prisma Studio for data inspection
npx prisma studio
```

## Key Implementation Details

### AI Service Integration
The chat service (`src/chat/chat.service.ts`) orchestrates complex AI interactions:
- Builds comprehensive conversation memory from session history and user topics
- Sends cultural profile and conversation context to FastAPI AI service
- Processes AI responses with emotional tone, topic detection, and cultural adaptations
- Updates user topics, progress tracking, and session insights based on AI analysis

### Cultural Awareness System
Cultural profiles use Zimbabwe-specific enums:
- **Age Groups**: YOUTH (13-24), ADULT (25-54), ELDER (55+)
- **Locations**: URBAN, RURAL, PERI_URBAN
- **Education**: PRIMARY, SECONDARY, TERTIARY, POSTGRADUATE
- **Family Types**: NUCLEAR, EXTENDED, SINGLE_PARENT, GUARDIAN
- **Respect Levels**: HIGH (traditional), MODERATE (balanced), RELAXED (modern)

### Memory Management
Sophisticated conversation memory tracks:
- Recent messages with emotional context
- Cross-session topic persistence with frequency and severity
- Session insights and action items
- Relationship strength and trust level progression

### API Architecture
- Global `/api` prefix for all endpoints
- Cookie-based authentication with CORS support
- Integration with external FastAPI service at localhost:8000
- Error handling with NestJS exception filters

## Development Workflow

### Starting the Platform
```bash
# Full platform with monitoring (recommended)
./start-calma.sh

# Quick development mode
./start-dev.sh

# Using npm scripts
npm run calma          # Full platform
npm run dev           # Development mode
npm run ai:start      # AI service only
npm run ai:health     # Check AI service
npm run integration:test  # Test integration
```

### Development Process
1. **Database Changes**: Update Prisma schema → `npx prisma generate` → `npx prisma db push`
2. **AI Service**: Will be started automatically by startup scripts
3. **Testing**: Run tests before committing changes
4. **Code Quality**: Use `npm run lint` and `npm run format` for consistent styling

### Startup Scripts
- `./start-calma.sh` - Full platform startup with health checks and monitoring
- `./start-dev.sh` - Simple development startup
- `calma-ai-service/start-ai-service.sh` - AI service only
- See `STARTUP.md` for detailed usage instructions

## FastAPI AI Service Integration

The backend integrates with a FastAPI microservice (`calma-ai-service/`) for AI inference:

### Integration Architecture
- **NestJS**: Handles business logic, database operations, user management, conversation memory
- **FastAPI**: Pure AI inference service with cultural awareness and analysis
- **Communication**: HTTP requests from NestJS to FastAPI `/infer` endpoint

### FastAPI Service Setup
```bash
# Start the AI service (must be running before NestJS)
cd calma-ai-service
pip install -r requirements.txt
cp .env.example .env  # Configure MODEL_PATH
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Integration Flow
1. **User sends message** → NestJS chat controller
2. **NestJS prepares context** → Cultural profile + conversation memory
3. **NestJS calls FastAPI** → `/infer` endpoint with message and context
4. **FastAPI analyzes & responds** → AI response + mood analysis + cultural elements
5. **NestJS processes results** → Updates database with insights and returns to frontend

### AI Service Features Integrated
- **Mood Detection**: Negative/neutral/positive with confidence scoring
- **Cultural Awareness**: Zimbabwean context with Ubuntu philosophy
- **Resource Recommendations**: Stress management, family counseling, financial guidance, etc.
- **Quality Metrics**: Empathy and cultural awareness scoring
- **Response Analysis**: Word count, cultural elements detected, emotional intensity

### Error Handling
- Service health checks before processing messages
- Graceful fallback for AI service unavailability
- Specific error messages for timeout, connection, and service issues
- Request timeout configuration (30 seconds)

### Health Monitoring
- **Chat Health Endpoint**: `GET /api/chat/health` - Checks AI service availability
- **FastAPI Health**: `GET localhost:8000/health` - Model status and service health
- **Integration Test**: Run `node test-integration.js` to verify end-to-end functionality

## External Dependencies

- **FastAPI AI Service**: Must be running on localhost:8000 for chat functionality
- **Fine-tuned Model**: Llama 3.2-3B with LoRA adapters in `models/calma-final/`
- **PostgreSQL**: Database connection configured via DATABASE_URL environment variable
- **Frontend CORS**: Configured for localhost:8080, localhost:5173, localhost:3001