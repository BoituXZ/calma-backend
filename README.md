# Calma Backend - Culturally-Aware Mental Health Platform

> A production-grade NestJS REST API providing secure, culturally-sensitive mental health chat services with AI integration, conversation memory management, and comprehensive data tracking.

[![NestJS](https://img.shields.io/badge/NestJS-E0234E?style=flat&logo=nestjs&logoColor=white)](https://nestjs.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Prisma](https://img.shields.io/badge/Prisma-2D3748?style=flat&logo=prisma&logoColor=white)](https://www.prisma.io/)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Design](#architecture--design)
- [Security Implementation](#security-implementation)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)

## Overview

Calma Backend is an enterprise-grade mental health platform backend designed with security, scalability, and cultural awareness at its core. The system demonstrates advanced backend engineering practices including:

- **Secure Authentication**: JWT-based authentication with HTTP-only cookies and role-based access control
- **Microservices Architecture**: Modular NestJS design with FastAPI AI service integration
- **Data Protection**: Comprehensive input validation, parameterized queries, and sensitive data handling
- **Scalable Database Design**: Sophisticated relational schema with conversation memory and cross-session tracking
- **Cultural Intelligence**: Zimbabwe-focused mental health support with localized contexts and cultural adaptations

This project showcases production-ready backend development skills essential for application security roles.

## Key Features

### 🔐 Security & Authentication
- **JWT Authentication** with secure HTTP-only cookie management
- **Role-based Access Control (RBAC)** with user/therapist/admin roles
- **Password Security** with bcrypt hashing (12 rounds)
- **CORS Configuration** with explicit origin whitelisting
- **Request Validation** using class-validator and DTOs
- **SQL Injection Prevention** via Prisma ORM parameterized queries
- **Session Management** with secure token rotation

### 🧠 AI Integration & Analysis
- **FastAPI Microservice Integration** for AI inference (Llama 3.2-3B fine-tuned model)
- **Conversation Memory System** with multi-session context persistence
- **Emotional Tone Detection** and sentiment analysis
- **Cultural Adaptation Engine** for Zimbabwe-specific mental health contexts
- **Resource Recommendation System** based on conversation analysis
- **Quality Metrics Tracking** (empathy scores, cultural awareness)

### 📊 Data Management & Tracking
- **Comprehensive User Profiles** with cultural demographics
- **Session-based Conversation Grouping** with AI-generated insights
- **Cross-session Topic Tracking** with severity and frequency monitoring
- **Progress Metrics** tracking relationship strength and trust levels
- **Mood Tracking System** with cultural factors integration
- **Resource Library** with cultural relevance scoring

### 🏗️ Architecture Best Practices
- **Modular Service Design** following SOLID principles
- **Dependency Injection** for testable, maintainable code
- **Repository Pattern** via Prisma service abstraction
- **Exception Filtering** with standardized error responses
- **Health Check Endpoints** for service monitoring
- **Graceful Error Handling** with fallback mechanisms

## Architecture & Design

### System Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │  HTTPS  │                  │  HTTP   │                 │
│  Frontend App   │◄───────►│  NestJS Backend  │◄───────►│  FastAPI AI     │
│  (React/Vue)    │         │  (Port 3000)     │         │  (Port 8000)    │
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │                            │
                                     │                            │
                                     ▼                            ▼
                            ┌──────────────────┐        ┌─────────────────┐
                            │   PostgreSQL     │        │   Fine-tuned    │
                            │   Database       │        │   Llama Model   │
                            │   (Prisma ORM)   │        │   (LoRA)        │
                            └──────────────────┘        └─────────────────┘
```

### Module Structure

```
src/
├── auth/                    # JWT authentication & authorization
│   ├── guards/             # Route protection (JWT, roles)
│   ├── strategies/         # Passport JWT strategy
│   └── dto/               # Auth data transfer objects
├── user/                   # User management & profiles
├── chat/                   # AI chatbot interaction & orchestration
├── therapist-chat/         # Human therapist communication
├── conversation-session/   # Session memory management
├── cultural-profile/       # Cultural context management
├── mood/                   # Mood tracking with cultural factors
├── resources/             # Mental health resource library
├── saved-resource/        # User's saved resources
├── database/              # Prisma service & database module
└── common/                # Shared interfaces & utilities
```

### Data Flow - Chat Message Processing

1. **Authentication Layer**: JWT validation → Role verification
2. **Controller**: Request validation → DTO transformation
3. **Service Orchestration**:
   - Fetch user's cultural profile
   - Build conversation memory (recent messages + user topics)
   - Prepare AI service request payload
4. **AI Integration**: HTTP request to FastAPI → Inference → Response parsing
5. **Data Processing**:
   - Extract emotional tone, detected topics, cultural adaptations
   - Update user topics with frequency and severity
   - Track progress metrics (relationship strength, trust level)
   - Generate session insights and action items
6. **Response**: Sanitized data → Client

## Security Implementation

### Authentication & Authorization

**JWT Token Management**
```typescript
// Secure cookie configuration
{
  httpOnly: true,      // Prevents XSS attacks
  secure: true,        // HTTPS only in production
  sameSite: 'strict',  // CSRF protection
  maxAge: 24 * 60 * 60 * 1000  // 24 hours
}
```

**Role-Based Access Control**
```typescript
@UseGuards(JwtAuthGuard, RolesGuard)
@Roles('admin', 'therapist')
```

### Input Validation

**Class-Validator DTOs**
```typescript
export class CreateChatDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(2000)
  message: string;

  @IsOptional()
  @IsUUID()
  sessionId?: string;
}
```

### Data Protection

- **Password Hashing**: bcrypt with 12 salt rounds
- **Parameterized Queries**: Prisma ORM prevents SQL injection
- **Sensitive Data Handling**: Passwords excluded from API responses
- **CORS Whitelist**: Explicit origin configuration
- **Environment Variables**: Secrets managed via `.env`

### API Security Headers

```typescript
app.enableCors({
  origin: ['http://localhost:8080', 'http://localhost:5173'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
});
```

### Error Handling

- **No Information Leakage**: Generic error messages to clients
- **Detailed Logging**: Internal error tracking for debugging
- **Graceful Degradation**: Fallback responses when AI service unavailable
- **Timeout Protection**: 30-second request timeouts

## Technology Stack

### Core Technologies
- **NestJS 10.x** - Progressive Node.js framework with TypeScript
- **TypeScript 5.x** - Type-safe development
- **PostgreSQL** - Relational database for complex data relationships
- **Prisma ORM** - Type-safe database client with migrations

### Authentication & Security
- **Passport JWT** - Token-based authentication
- **bcrypt** - Password hashing
- **class-validator** - Input validation
- **class-transformer** - DTO transformation

### External Services
- **FastAPI** - AI inference microservice (Python)
- **Llama 3.2-3B** - Fine-tuned language model with LoRA adapters

### Development Tools
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **Jest** - Unit and integration testing
- **Prisma Studio** - Database management GUI

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- PostgreSQL 14.x or higher
- Python 3.9+ (for AI service)
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/calma-backend.git
cd calma-backend
```

2. **Install dependencies**
```bash
npm install
```

3. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
DATABASE_URL="postgresql://user:password@localhost:5432/calma"
JWT_SECRET="your-secure-jwt-secret-minimum-32-characters"
JWT_EXPIRES_IN="24h"
AI_SERVICE_URL="http://localhost:8000"
NODE_ENV="development"
```

4. **Set up the database**
```bash
# Generate Prisma client
npx prisma generate

# Run migrations
npx prisma migrate dev

# (Optional) Seed database
npm run seed
```

5. **Start the AI service**
```bash
cd calma-ai-service
pip install -r requirements.txt
cp .env.example .env
# Configure MODEL_PATH in .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Start the backend**
```bash
# Development mode with hot reload
npm run start:dev

# Or use the integrated startup script
./start-calma.sh
```

The API will be available at `http://localhost:3000/api`

### Quick Start Scripts

```bash
# Full platform startup with monitoring
./start-calma.sh

# Development mode only
./start-dev.sh

# Check system health
npm run ai:health
npm run integration:test
```

## API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

Returns JWT token in HTTP-only cookie.

#### Logout
```http
POST /api/auth/logout
Authorization: Bearer <token>
```

### Chat Endpoints

#### Send Message
```http
POST /api/chat
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "I've been feeling stressed about work",
  "sessionId": "uuid-optional"
}
```

Response:
```json
{
  "id": "uuid",
  "message": "I've been feeling stressed about work",
  "response": "I understand work stress can be overwhelming...",
  "emotionalTone": "concerned",
  "detectedTopics": ["work_stress", "anxiety"],
  "culturalAdaptations": ["ubuntu_philosophy", "community_support"],
  "timestamp": "2025-10-13T10:30:00Z",
  "sessionId": "uuid"
}
```

#### Get Conversation History
```http
GET /api/chat/history?limit=50&offset=0
Authorization: Bearer <token>
```

#### Health Check
```http
GET /api/chat/health
```

### Cultural Profile Endpoints

#### Create/Update Profile
```http
POST /api/cultural-profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "ageGroup": "ADULT",
  "location": "URBAN",
  "primaryLanguage": "SHONA",
  "education": "TERTIARY",
  "ethnicity": "Shona",
  "religion": "Christian",
  "familyStructure": "EXTENDED",
  "respectLevel": "MODERATE",
  "economicStatus": "MIDDLE_CLASS"
}
```

### Session Endpoints

#### Get Active Session
```http
GET /api/conversation-session/active
Authorization: Bearer <token>
```

#### Get Session Details
```http
GET /api/conversation-session/:id
Authorization: Bearer <token>
```

### Additional Endpoints

- **Mood Tracking**: `/api/mood`
- **Resources**: `/api/resources`
- **Saved Resources**: `/api/saved-resource`
- **User Profile**: `/api/user/profile`
- **Therapist Chat**: `/api/therapist-chat`

Full API documentation available via Swagger at `/api/docs` (when enabled).

## Database Schema

### Core Models

**User**
- Authentication (email, password hash)
- Profile information
- Role-based access (user/therapist/admin)
- Relationships: chats, sessions, topics, mood entries

**Chat**
- Message content and AI response
- Emotional tone detection
- Detected topics array
- Cultural adaptations applied
- Quality metrics (empathy score, cultural awareness)
- Session association

**CulturalProfile**
- Demographic information (age, location, education)
- Cultural context (ethnicity, religion, language)
- Family structure and respect level
- Economic status
- One-to-one with User

**ConversationSession**
- Session grouping for conversation memory
- AI-generated session insights
- Action items tracking
- Topic summaries
- Start/end timestamps
- Active status

**UserTopic**
- Cross-session topic persistence
- Frequency and severity tracking
- Status monitoring (new/ongoing/resolved)
- First and last mentioned timestamps
- Topic descriptions

**ProgressTracking**
- Relationship strength metrics
- Trust level progression
- Improvement indicators
- Timestamp-based tracking
- Session associations

### Enums (Zimbabwe-specific)

```typescript
enum AgeGroup { YOUTH, ADULT, ELDER }
enum Location { URBAN, RURAL, PERI_URBAN }
enum PrimaryLanguage { SHONA, NDEBELE, ENGLISH }
enum EducationLevel { PRIMARY, SECONDARY, TERTIARY, POSTGRADUATE }
enum FamilyStructure { NUCLEAR, EXTENDED, SINGLE_PARENT, GUARDIAN }
enum RespectLevel { HIGH, MODERATE, RELAXED }
enum EconomicStatus { LOW, LOWER_MIDDLE, MIDDLE_CLASS, UPPER_MIDDLE, HIGH }
```

### Database Diagram

See [`src/database/prisma/schema.prisma`](src/database/prisma/schema.prisma) for complete schema definition.

## Development

### Project Structure

```
calma-backend/
├── src/
│   ├── auth/              # Authentication module
│   ├── user/              # User management
│   ├── chat/              # Chat service (AI integration)
│   ├── conversation-session/  # Session management
│   ├── cultural-profile/  # Cultural context
│   ├── mood/              # Mood tracking
│   ├── resources/         # Resource library
│   ├── saved-resource/    # Saved resources
│   ├── therapist-chat/    # Therapist communication
│   ├── database/          # Prisma service
│   ├── common/            # Shared utilities
│   └── main.ts            # Application entry point
├── calma-ai-service/      # FastAPI AI microservice
├── prisma/
│   └── schema.prisma      # Database schema
├── test/                  # E2E tests
├── .env.example           # Environment template
├── start-calma.sh         # Full platform startup
├── start-dev.sh           # Development startup
└── CLAUDE.md              # Development guidance
```

### Development Workflow

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes with type safety**
```bash
# TypeScript compiler checks
npm run build

# Linting
npm run lint

# Auto-fix linting issues
npm run lint -- --fix

# Format code
npm run format
```

3. **Update database schema**
```bash
# Edit prisma/schema.prisma
npx prisma generate
npx prisma migrate dev --name descriptive_migration_name
```

4. **Write tests**
```bash
# Unit tests
npm run test

# Watch mode
npm run test:watch

# E2E tests
npm run test:e2e

# Coverage report
npm run test:cov
```

5. **Test integration**
```bash
# Start AI service
cd calma-ai-service && uvicorn app.main:app --reload

# In another terminal, start backend
npm run start:dev

# Test integration
node test-integration.js
```

6. **Commit and push**
```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

### Code Quality Standards

- **TypeScript strict mode** enabled
- **ESLint** for code quality
- **Prettier** for consistent formatting
- **Conventional Commits** for clear history
- **100% DTO validation** for all inputs
- **Unit test coverage** target: >80%

### Environment Variables

```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/calma"

# Authentication
JWT_SECRET="minimum-32-character-secret-key-here"
JWT_EXPIRES_IN="24h"

# External Services
AI_SERVICE_URL="http://localhost:8000"
AI_SERVICE_TIMEOUT="30000"

# Application
NODE_ENV="development"
PORT="3000"

# CORS Origins (comma-separated)
CORS_ORIGINS="http://localhost:8080,http://localhost:5173,http://localhost:3001"

# Logging
LOG_LEVEL="debug"
```

## Testing

### Unit Tests

```bash
# Run all unit tests
npm run test

# Watch mode for TDD
npm run test:watch

# Coverage report
npm run test:cov

# Debug tests
npm run test:debug
```

### End-to-End Tests

```bash
# Run E2E tests
npm run test:e2e

# E2E with coverage
npm run test:e2e:cov
```

### Integration Tests

```bash
# Test AI service integration
node test-integration.js

# Check AI service health
npm run ai:health
```

### Testing Best Practices

- **Mock external services** (AI service, database)
- **Test authentication guards** and role permissions
- **Validate input sanitization** and DTO transformations
- **Test error handling** and edge cases
- **Integration tests** for critical paths

### Example Test

```typescript
describe('ChatService', () => {
  it('should sanitize user input before AI processing', async () => {
    const maliciousInput = '<script>alert("xss")</script>';
    const result = await chatService.sendMessage(userId, maliciousInput);

    expect(result.message).not.toContain('<script>');
    expect(result.response).toBeDefined();
  });

  it('should handle AI service unavailability gracefully', async () => {
    jest.spyOn(httpService, 'post').mockImplementation(() =>
      throwError(() => new Error('Service unavailable'))
    );

    await expect(
      chatService.sendMessage(userId, 'test message')
    ).rejects.toThrow('AI service temporarily unavailable');
  });
});
```

## Deployment

### Production Build

```bash
# Build the application
npm run build

# Run production server
npm run start:prod
```

### Environment Configuration

**Production environment variables:**
```env
NODE_ENV="production"
DATABASE_URL="postgresql://user:password@prod-db:5432/calma"
JWT_SECRET="<strong-random-secret-minimum-32-chars>"
AI_SERVICE_URL="http://ai-service:8000"
CORS_ORIGINS="https://yourdomain.com"
```

### Database Migrations

```bash
# Generate migration
npx prisma migrate dev --name migration_name

# Deploy to production
npx prisma migrate deploy

# Prisma client generation
npx prisma generate
```

### Docker Deployment (Example)

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npx prisma generate
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start:prod"]
```

### Health Checks

**Backend Health**
```bash
curl http://localhost:3000/api/health
```

**AI Service Health**
```bash
curl http://localhost:8000/health
```

**Database Health**
```bash
npx prisma db pull
```

### Security Considerations for Production

- [ ] Enable HTTPS/TLS for all communications
- [ ] Use environment-specific JWT secrets (rotate regularly)
- [ ] Implement rate limiting (e.g., express-rate-limit)
- [ ] Enable CORS only for trusted domains
- [ ] Set secure cookie flags (httpOnly, secure, sameSite)
- [ ] Implement request logging and monitoring
- [ ] Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- [ ] Enable database connection pooling
- [ ] Implement backup and disaster recovery
- [ ] Set up monitoring and alerting (e.g., Datadog, New Relic)

### Monitoring & Logging

```typescript
// Winston logger configuration (example)
{
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
}
```

## Performance Optimization

- **Database indexing** on frequently queried fields
- **Connection pooling** for PostgreSQL
- **Caching strategy** for cultural profiles and resources
- **Lazy loading** for relationships in Prisma
- **Request timeout** configuration (30s default)
- **Payload size limits** to prevent DoS

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `security:` - Security enhancements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

## Acknowledgments

- NestJS framework for excellent TypeScript backend architecture
- Prisma for type-safe database access
- FastAPI for high-performance AI inference
- The open-source community for inspiration and tools

---

**Note**: This is a portfolio project demonstrating production-grade backend development practices, security implementation, and full-stack integration skills. The mental health context is used to showcase handling of sensitive data and culturally-aware application design.

For questions or collaboration opportunities, please reach out via [email@example.com](mailto:email@example.com).
