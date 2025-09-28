#!/bin/bash

# Simple Development Startup Script for Calma
# This is a lightweight version for quick development

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Calma Development Environment...${NC}"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Must run from calma-backend directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🔄 Stopping services...${NC}"
    pkill -f "uvicorn.*app.main" 2>/dev/null || true
    pkill -f "nest.*start" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start AI service in background
echo -e "${YELLOW}🤖 Starting AI service...${NC}"
cd calma-ai-service
python -m app.main > /tmp/ai-service.log 2>&1 &
cd ..

# Wait a bit for AI service
echo -e "${YELLOW}⏳ Waiting for AI service (30s)...${NC}"
sleep 30

# Check if AI service is ready
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ AI service ready${NC}"
else
    echo -e "${YELLOW}⚠️  AI service still loading...${NC}"
fi

# Start NestJS
echo -e "${YELLOW}🏗️  Starting NestJS...${NC}"
npm run start:dev