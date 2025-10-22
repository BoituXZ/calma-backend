# Calma Backend API Documentation

**Base URL:** `http://localhost:3000/api`

**Authentication:** Most endpoints require JWT authentication via HTTP-only cookies. After login/signup, a JWT cookie is automatically set.

---

## 🔐 Authentication Endpoints

### POST `/api/auth/signup`
Register a new user account.

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securePassword123",
  "role": "USER"
}
```

**Fields:**
- `name` (required): User's full name
- `email` (required): Valid email address
- `password` (required): Minimum 8 characters
- `role` (optional): Either "USER" or "THERAPIST" (defaults to "USER" if not provided)

**Note:** Cannot create ADMIN accounts via signup endpoint for security reasons.

**Response:**
```json
{
  "message": "User created successfully",
  "user": {
    "id": "uuid",
    "name": "John Doe",
    "email": "john@example.com",
    "role": "USER"
  }
}
```

**Note:** Sets JWT cookie automatically.

---

### POST `/api/auth/login`
Login to existing account.

**Request Body:**
```json
{
  "email": "john@example.com",
  "password": "securePassword123"
}
```

**Response:**
```json
{
  "message": "Logged in successfully",
  "user": {
    "id": "uuid",
    "name": "John Doe",
    "email": "john@example.com",
    "role": "USER"
  }
}
```

**Note:** Sets JWT cookie automatically.

---

### POST `/api/auth/logout`
Logout current user.

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

---

### GET `/api/auth/me`
Get current authenticated user.

**Response:**
```json
{
  "user": {
    "id": "uuid",
    "name": "John Doe",
    "email": "john@example.com",
    "role": "USER"
  }
}
```

---

## 👤 User Profile Endpoints

**All endpoints require authentication**

### GET `/api/user/profile`
Get own user profile.

**Response:**
```json
{
  "id": "uuid",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "USER",
  "createdAt": "2024-01-01T00:00:00.000Z"
}
```

---

### PUT `/api/user/profile`
Update own profile.

**Request Body:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "Jane Doe",
  "email": "jane@example.com",
  "role": "USER",
  "createdAt": "2024-01-01T00:00:00.000Z"
}
```

---

### DELETE `/api/user/profile`
Delete own account.

**Response:**
```json
{
  "message": "User deleted successfully"
}
```

---

### GET `/api/user/therapists`
Get list of all therapists (available to all authenticated users).

**Response:**
```json
{
  "therapists": [
    {
      "id": "uuid",
      "name": "Dr. Jane Smith",
      "email": "jane.smith@example.com",
      "role": "THERAPIST",
      "createdAt": "2024-01-01T00:00:00.000Z"
    },
    {
      "id": "uuid",
      "name": "Dr. John Doe",
      "email": "john.doe@example.com",
      "role": "THERAPIST",
      "createdAt": "2024-01-01T00:00:00.000Z"
    }
  ]
}
```

**Note:** Results are sorted alphabetically by name.

---

## 💬 AI Chat Endpoints

**All endpoints require authentication**

### GET `/api/chat/health`
Check if AI service is available.

**Response:**
```json
{
  "status": "healthy",
  "ai_service": "available",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

---

### POST `/api/chat/message`
Send a message to the AI chatbot.

**Request Body:**
```json
{
  "message": "I'm feeling stressed about work",
  "sessionId": "optional-session-uuid"
}
```

**Response:**
```json
{
  "userMessage": {
    "id": "uuid",
    "message": "I'm feeling stressed about work",
    "sender": "USER",
    "timestamp": "2024-01-01T00:00:00.000Z"
  },
  "botMessage": {
    "id": "uuid",
    "message": "I understand work can be stressful...",
    "sender": "BOT",
    "timestamp": "2024-01-01T00:00:00.000Z",
    "emotionalTone": "anxious",
    "detectedTopics": ["work_stress", "stress_management"],
    "analysisResults": {
      "mood_detected": "negative",
      "confidence": 0.85,
      "emotional_intensity": 7,
      "suggested_resources": ["stress_management", "work_life_balance"],
      "cultural_elements_detected": ["ubuntu", "community_support"],
      "quality_metrics": {
        "empathy_score": 0.9,
        "cultural_awareness_score": 0.85
      }
    }
  },
  "session": {
    "id": "session-uuid",
    "title": "Work Stress Discussion"
  }
}
```

---

### GET `/api/chat/history`
Get chat history.

**Query Parameters:**
- `limit` (optional): Number of messages to retrieve (default: 50)
- `sessionId` (optional): Get chats for specific session

**Response (without sessionId):**
```json
{
  "chats": [
    {
      "id": "uuid",
      "message": "Hello",
      "sender": "USER",
      "timestamp": "2024-01-01T00:00:00.000Z",
      "emotionalTone": null,
      "detectedTopics": []
    }
  ]
}
```

**Response (with sessionId):**
```json
{
  "session": {
    "id": "session-uuid",
    "title": "Session Title",
    "startTime": "2024-01-01T00:00:00.000Z",
    "endTime": null
  },
  "chats": [...]
}
```

---

### GET `/api/chat/sessions`
Get all conversation sessions for current user.

**Response:**
```json
{
  "sessions": [
    {
      "id": "uuid",
      "title": "Work Stress Discussion",
      "startTime": "2024-01-01T00:00:00.000Z",
      "endTime": null,
      "primaryTopic": "work_stress",
      "emotionalState": "anxious",
      "followUpNeeded": false,
      "_count": {
        "chats": 12
      }
    }
  ]
}
```

---

## 🧠 Cultural Profile Endpoints

**All endpoints require authentication**

### POST `/api/cultural-profile/:userId`
Create cultural profile.

**Request Body:**
```json
{
  "ageGroup": "ADULT",
  "location": "URBAN",
  "educationLevel": "TERTIARY",
  "ethnicBackground": "Shona",
  "religiousBackground": "Christian",
  "languagePreference": "English",
  "familyStructure": "NUCLEAR",
  "householdSize": 4,
  "hasElders": true,
  "communicationStyle": "direct",
  "respectLevel": "MODERATE",
  "economicStatus": "MIDDLE",
  "employmentStatus": "employed"
}
```

**Enums:**
- `ageGroup`: YOUTH, ADULT, ELDER
- `location`: URBAN, RURAL, PERI_URBAN
- `educationLevel`: PRIMARY, SECONDARY, TERTIARY, POSTGRADUATE
- `familyStructure`: NUCLEAR, EXTENDED, SINGLE_PARENT, GUARDIAN
- `respectLevel`: HIGH, MODERATE, RELAXED
- `economicStatus`: LOW, MIDDLE, HIGH

---

### GET `/api/cultural-profile/:userId`
Get cultural profile.

---

### PUT `/api/cultural-profile/:userId`
Update cultural profile (same body as POST).

---

### POST `/api/cultural-profile/:userId/default`
Create default profile with standard values.

---

## 📊 Mood Tracking Endpoints

**All endpoints require authentication**

### POST `/api/mood/moods`
Add mood entry.

**Request Body:**
```json
{
  "id": "user-uuid",
  "mood": 3
}
```

**Mood Values:**
- 1: Very Low
- 2: Low
- 3: Neutral
- 4: Good
- 5: Very Good

**Response:**
```json
{
  "message": "Mood added"
}
```

---

### GET `/api/mood/moods?userId={userId}`
Get user's mood history.

**Response:**
```json
{
  "message": [
    { "mood": "Good" },
    { "mood": "Neutral" }
  ]
}
```

---

## 📚 Resources Endpoints

### GET `/api/resources`
Get all mental health resources (public).

**Query Parameters:**
- `type` (optional): VIDEO, ARTICLE, TOOL, PODCAST, CULTURAL_STORY
- `tags` (optional): Comma-separated tags (e.g., "stress,anxiety")
- `culturalTags` (optional): Comma-separated cultural tags (e.g., "zimbabwean,shona")

**Response:**
```json
{
  "resources": [
    {
      "id": "uuid",
      "title": "Managing Stress",
      "description": "A guide to stress management",
      "type": "ARTICLE",
      "link": "https://example.com/article",
      "tags": ["stress", "anxiety"],
      "culturalTags": ["zimbabwean", "urban"],
      "targetAudience": ["youth", "adult"],
      "createdAt": "2024-01-01T00:00:00.000Z"
    }
  ]
}
```

---

### GET `/api/resources/:id`
Get single resource by ID.

---

### POST `/api/resources`
Create new resource (requires THERAPIST or ADMIN role).

**Request Body:**
```json
{
  "title": "Managing Stress",
  "description": "A guide to stress management",
  "type": "ARTICLE",
  "link": "https://example.com/article",
  "tags": ["stress", "anxiety"],
  "culturalTags": ["zimbabwean"],
  "targetAudience": ["youth"]
}
```

---

## ⭐ Saved Resources Endpoints

**All endpoints require authentication**

### POST `/api/saved-resource`
Save a resource to user's collection.

**Request Body:**
```json
{
  "resourceId": "resource-uuid",
  "recommendationReason": "Helpful for stress management",
  "culturalRelevance": "Includes Zimbabwean context"
}
```

**Response:**
```json
{
  "savedResource": {
    "id": "uuid",
    "userId": "user-uuid",
    "resourceId": "resource-uuid",
    "savedAt": "2024-01-01T00:00:00.000Z",
    "resource": { ... }
  },
  "message": "Resource saved successfully"
}
```

---

### GET `/api/saved-resource`
Get all saved resources for current user.

**Response:**
```json
{
  "savedResources": [
    {
      "id": "uuid",
      "savedAt": "2024-01-01T00:00:00.000Z",
      "recommendationReason": "Helpful for stress",
      "resource": { ... }
    }
  ]
}
```

---

### DELETE `/api/saved-resource/:id`
Remove saved resource.

**Response:**
```json
{
  "message": "Resource removed from saved"
}
```

---

## 🏥 Therapist Chat Endpoints

**All endpoints require authentication**

### POST `/api/therapist-chat/message`
User sends message to therapist.

**Request Body:**
```json
{
  "therapistId": "therapist-uuid",
  "message": "I need help with anxiety"
}
```

**Response:**
```json
{
  "message": {
    "id": "uuid",
    "userId": "user-uuid",
    "therapistId": "therapist-uuid",
    "message": "I need help with anxiety",
    "sender": "USER",
    "timestamp": "2024-01-01T00:00:00.000Z"
  },
  "success": true
}
```

---

### POST `/api/therapist-chat/therapist/message`
Therapist sends message to user (requires THERAPIST role).

**Request Body:**
```json
{
  "userId": "user-uuid",
  "message": "I can help you with that"
}
```

---

### GET `/api/therapist-chat/conversation/:therapistId`
Get conversation with specific therapist.

**Response:**
```json
{
  "messages": [
    {
      "id": "uuid",
      "message": "Hello",
      "sender": "USER",
      "timestamp": "2024-01-01T00:00:00.000Z",
      "user": {
        "id": "uuid",
        "name": "John Doe",
        "email": "john@example.com"
      }
    }
  ]
}
```

---

### GET `/api/therapist-chat/conversations`
Get all user's conversations.

**Response:**
```json
{
  "conversations": [
    {
      "therapist": {
        "id": "uuid",
        "name": "Dr. Smith",
        "email": "smith@example.com"
      },
      "lastMessage": {
        "message": "Let's schedule a session",
        "timestamp": "2024-01-01T00:00:00.000Z"
      },
      "messageCount": 15
    }
  ]
}
```

---

### GET `/api/therapist-chat/therapist/conversations`
Get all therapist's conversations (requires THERAPIST role).

---

## 📅 Appointments Endpoints

**All endpoints require authentication**

### POST `/api/appointments`
Create new appointment.

**Request Body:**
```json
{
  "therapistId": "therapist-uuid",
  "scheduledAt": "2024-01-15T10:00:00.000Z",
  "duration": 60,
  "reason": "Anxiety counseling",
  "notes": "First session",
  "meetingLink": "https://zoom.us/j/123",
  "location": "Office Room 3"
}
```

**Response:**
```json
{
  "appointment": {
    "id": "uuid",
    "userId": "user-uuid",
    "therapistId": "therapist-uuid",
    "scheduledAt": "2024-01-15T10:00:00.000Z",
    "duration": 60,
    "status": "SCHEDULED",
    "reason": "Anxiety counseling",
    "user": { "id": "...", "name": "...", "email": "..." }
  },
  "message": "Appointment created successfully"
}
```

**Status Values:**
- SCHEDULED
- CONFIRMED
- CANCELLED
- COMPLETED
- NO_SHOW

---

### GET `/api/appointments/user`
Get all user's appointments.

**Response:**
```json
{
  "appointments": [
    {
      "id": "uuid",
      "scheduledAt": "2024-01-15T10:00:00.000Z",
      "duration": 60,
      "status": "SCHEDULED",
      "reason": "Anxiety counseling"
    }
  ]
}
```

---

### GET `/api/appointments/therapist`
Get all therapist's appointments (requires THERAPIST role).

---

### GET `/api/appointments/:id`
Get single appointment by ID.

---

### PUT `/api/appointments/:id`
Update appointment.

**Request Body:**
```json
{
  "scheduledAt": "2024-01-15T14:00:00.000Z",
  "status": "CONFIRMED",
  "notes": "Updated notes"
}
```

---

### DELETE `/api/appointments/:id`
Cancel appointment.

**Response:**
```json
{
  "appointment": { ... },
  "message": "Appointment cancelled successfully"
}
```

---

## 🔒 Role-Based Access Control

### Roles:
- **USER**: Default role, access to chat, resources, profile, appointments
- **THERAPIST**: Can message users, view their appointments
- **ADMIN**: Full access to all endpoints and user management

### Protected Endpoints:
- Therapist-only endpoints marked with "(requires THERAPIST role)"
- Admin-only endpoints marked with "(requires ADMIN role)"

---

## 🍪 Cookie Configuration

**Frontend CORS Setup Required:**
```javascript
fetch('http://localhost:3000/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  credentials: 'include', // IMPORTANT: Include cookies
  body: JSON.stringify({ email, password })
})
```

**Allowed Origins:**
- http://localhost:8080
- http://localhost:5173
- http://localhost:3001

---

## 📝 Error Responses

All endpoints return standardized error responses:

```json
{
  "statusCode": 400,
  "message": "Error description",
  "error": "Bad Request"
}
```

**Common Status Codes:**
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized (not logged in)
- 403: Forbidden (insufficient permissions)
- 404: Not Found
- 409: Conflict (e.g., duplicate resource)
- 500: Internal Server Error

---

## 🚀 Getting Started

1. **Base URL**: `http://localhost:3000/api`
2. **Authentication**: Use `/api/auth/signup` or `/api/auth/login`
3. **CORS**: Always include `credentials: 'include'` in fetch requests
4. **Headers**: Set `Content-Type: application/json` for POST/PUT requests

---

## 🔄 Implementation Status

### ✅ Fully Implemented:
- Authentication (signup, login, logout, me)
- User profile management
- AI chat with mood detection
- Chat history and sessions
- Cultural profiling
- Mood tracking
- Resources (CRUD)
- Saved resources
- Therapist chat
- Appointments system
- Role-based access control

### ⚠️ Partially Implemented:
- Onboarding questionnaire (model exists, no dedicated endpoints)
- Self-assessment (model exists, no dedicated endpoints)
- GDPR compliance features (basic user deletion exists)

---

## 📞 Need Help?

For questions or issues, check:
- Project README: `/README.md`
- Claude instructions: `/CLAUDE.md`
- Startup guide: `/STARTUP.md`
