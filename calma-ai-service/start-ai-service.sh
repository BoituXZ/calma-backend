#!/bin/bash

# Calma AI Service Startup Script
# This script starts the FastAPI AI inference service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Starting Calma AI Inference Service...${NC}"

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo -e "${RED}❌ Error: Must run this script from the calma-ai-service directory${NC}"
    echo -e "${YELLOW}💡 Run: cd calma-ai-service && ./start-ai-service.sh${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
        echo -e "${YELLOW}💡 Please review .env file and adjust MODEL_PATH if needed${NC}"
    else
        echo -e "${RED}❌ Error: No .env.example found${NC}"
        exit 1
    fi
fi

# Check if model directory exists
MODEL_PATH=$(grep "^MODEL_PATH=" .env | cut -d'=' -f2 | tr -d '"')
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Model directory not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}💡 Please check MODEL_PATH in .env file${NC}"
    echo -e "${YELLOW}💡 Expected structure: $MODEL_PATH/adapter_config.json${NC}"
    exit 1
fi

# Check if model files exist
if [ ! -f "$MODEL_PATH/adapter_config.json" ]; then
    echo -e "${RED}❌ Error: adapter_config.json not found in $MODEL_PATH${NC}"
    echo -e "${YELLOW}💡 Please ensure your fine-tuned model is in the correct location${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Model files found in: $MODEL_PATH${NC}"

# Check if Python dependencies are installed
echo -e "${BLUE}🔍 Checking dependencies...${NC}"
if ! python -c "import fastapi, uvicorn, transformers, torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing missing dependencies...${NC}"
    pip install -r requirements.txt
fi

echo -e "${GREEN}✅ Dependencies verified${NC}"

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
    echo -e "${YELLOW}💡 Checking if it's already our service...${NC}"

    # Test if it's our service
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Calma AI service is already running!${NC}"
        echo -e "${BLUE}🌐 Service URL: http://localhost:8000${NC}"
        echo -e "${BLUE}📚 API Docs: http://localhost:8000/docs${NC}"
        echo -e "${BLUE}❤️  Health Check: http://localhost:8000/health${NC}"
        exit 0
    else
        echo -e "${RED}❌ Error: Port 8000 is occupied by another service${NC}"
        echo -e "${YELLOW}💡 Please stop the other service or change the port in .env${NC}"
        exit 1
    fi
fi

# Start the service
echo -e "${BLUE}🚀 Starting FastAPI server...${NC}"
echo -e "${YELLOW}📝 This may take 30-60 seconds to load the model...${NC}"

# Create a temporary file to track startup
STARTUP_LOG="/tmp/calma-ai-startup.log"

# Start the service in background and capture logs
python -m app.main > "$STARTUP_LOG" 2>&1 &
SERVICE_PID=$!

echo -e "${BLUE}🔄 Loading model... (PID: $SERVICE_PID)${NC}"

# Wait for service to be ready (check health endpoint)
MAX_WAIT=120  # 2 minutes
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))

    # Show progress
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo -e "${YELLOW}⏳ Still loading... (${WAIT_COUNT}s/${MAX_WAIT}s)${NC}"
    fi

    # Check if process is still running
    if ! kill -0 $SERVICE_PID 2>/dev/null; then
        echo -e "${RED}❌ Service process died. Check logs:${NC}"
        tail -n 20 "$STARTUP_LOG"
        exit 1
    fi

    # Check if service is ready
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        break
    fi
done

# Check final status
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}🎉 Calma AI service started successfully!${NC}"
    echo -e "${GREEN}✅ Model loaded and ready for inference${NC}"
    echo ""
    echo -e "${BLUE}📊 Service Information:${NC}"
    echo -e "${BLUE}🌐 Service URL: http://localhost:8000${NC}"
    echo -e "${BLUE}📚 API Documentation: http://localhost:8000/docs${NC}"
    echo -e "${BLUE}❤️  Health Check: http://localhost:8000/health${NC}"
    echo -e "${BLUE}🤖 Model Info: http://localhost:8000/model-info${NC}"
    echo ""
    echo -e "${YELLOW}💡 To stop the service: kill $SERVICE_PID${NC}"
    echo -e "${YELLOW}📝 Service logs: tail -f $STARTUP_LOG${NC}"
    echo ""
    echo -e "${GREEN}🔗 Ready for NestJS integration!${NC}"

    # Keep the script running to show that service is active
    wait $SERVICE_PID
else
    echo -e "${RED}❌ Failed to start service within ${MAX_WAIT} seconds${NC}"
    echo -e "${RED}🔍 Check logs for details:${NC}"
    tail -n 20 "$STARTUP_LOG"

    # Kill the process if it's still running
    if kill -0 $SERVICE_PID 2>/dev/null; then
        kill $SERVICE_PID
    fi
    exit 1
fi