# 🚀 Calma Platform Startup Guide

This guide explains how to start the Calma mental health platform with its integrated AI service.

## 📋 Prerequisites

- **Node.js** (v16 or higher)
- **Python 3.8+**
- **PostgreSQL** database running
- **Fine-tuned model** in `calma-ai/models/calma-final/`

## 🎯 Quick Start Options

### Option 1: Full Platform (Recommended)
```bash
# Start both AI service and NestJS backend with monitoring
./start-calma.sh
```

### Option 2: Simple Development
```bash
# Lightweight version for quick development
./start-dev.sh
```

### Option 3: Using npm scripts
```bash
# Start full platform
npm run calma

# Start development mode
npm run dev

# Start only AI service
npm run ai:start

# Check AI service health
npm run ai:health

# Test integration
npm run integration:test
```

## 🔧 Manual Setup

### 1. Start AI Service Only
```bash
cd calma-ai-service
./start-ai-service.sh
```

### 2. Start NestJS Backend Only
```bash
npm run start:dev
```

## 📊 Service Information

### FastAPI AI Service
- **URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs
- **Model Info**: http://localhost:8000/model-info

### NestJS Backend API
- **URL**: http://localhost:3000
- **Chat Health**: http://localhost:3000/api/chat/health
- **API Base**: http://localhost:3000/api

## 🔍 Troubleshooting

### AI Service Issues

**Model not loading:**
```bash
# Check model path in calma-ai-service/.env
MODEL_PATH=../calma-ai/models/calma-final

# Verify model files exist
ls -la calma-ai/models/calma-final/
# Should contain: adapter_config.json, adapter_model.safetensors
```

**Port 8000 already in use:**
```bash
# Kill existing process
pkill -f "uvicorn.*app.main"

# Or change port in calma-ai-service/.env
PORT=8001
```

**Slow inference (CPU mode):**
- AI service is running on CPU (no GPU detected)
- First inference may take 30+ seconds
- Subsequent requests should be faster
- Consider increasing timeout in config

### Backend Issues

**Database connection:**
```bash
# Check DATABASE_URL in .env
DATABASE_URL="postgresql://user:password@localhost:5432/calma?schema=public"

# Run Prisma commands
npx prisma generate
npx prisma db push
```

**Port 3000 already in use:**
```bash
# Kill existing process
pkill -f "nest.*start"

# Or change port in main.ts
const port = process.env.PORT || 3001;
```

## 📝 Log Files

When using the startup scripts, logs are saved to:
- **AI Service**: `/tmp/calma-ai-service.log`
- **NestJS**: `/tmp/calma-nestjs.log`

View logs in real-time:
```bash
# AI service logs
tail -f /tmp/calma-ai-service.log

# NestJS logs
tail -f /tmp/calma-nestjs.log
```

## 🔄 Stopping Services

**Using startup scripts:**
- Press `Ctrl+C` in the terminal running the script

**Manual cleanup:**
```bash
# Kill AI service
pkill -f "uvicorn.*app.main"

# Kill NestJS
pkill -f "nest.*start"

# Kill all node processes (if needed)
pkill node
```

## 🧪 Testing the Integration

### Health Checks
```bash
# Test AI service
curl http://localhost:8000/health

# Test NestJS integration
curl http://localhost:3000/api/chat/health

# Run integration test
npm run integration:test
```

### Sample Chat Request
```bash
curl -X POST http://localhost:3000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "your-user-uuid",
    "message": "I am feeling stressed about my studies"
  }'
```

## ⚙️ Configuration

### AI Service Configuration
Edit `calma-ai-service/.env`:
```bash
MODEL_PATH=../calma-ai/models/calma-final
PORT=8000
TEMPERATURE=0.8
MAX_TOKENS=256
INFERENCE_TIMEOUT=30
```

### Backend Configuration
Edit `.env`:
```bash
DATABASE_URL="postgresql://user:password@localhost:5432/calma"
PORT=3000
```

## 🎯 Development Tips

1. **First-time setup**: Use `./start-calma.sh` for comprehensive startup with checks
2. **Daily development**: Use `./start-dev.sh` for faster startup
3. **AI-only testing**: Use `npm run ai:start` to test AI service separately
4. **Quick health check**: Use `npm run ai:health` to verify AI service status

## 📋 Startup Checklist

- [ ] PostgreSQL database is running
- [ ] Model files exist in `calma-ai/models/calma-final/`
- [ ] `.env` file configured in both root and `calma-ai-service/`
- [ ] npm dependencies installed (`npm install`)
- [ ] Python dependencies installed (`pip install -r calma-ai-service/requirements.txt`)
- [ ] Ports 3000 and 8000 are available

## 🆘 Getting Help

If you encounter issues:

1. **Check logs** in `/tmp/calma-*.log`
2. **Verify prerequisites** are installed
3. **Test components separately** (AI service first, then backend)
4. **Check port availability** (`lsof -i :8000` and `lsof -i :3000`)
5. **Review configuration** in `.env` files