#!/bin/bash

# Calma Backend Startup Script
# This script starts both the FastAPI AI service and NestJS backend

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Header
echo -e "${CYAN}"
echo "  ██████╗ █████╗ ██╗     ███╗   ███╗ █████╗ "
echo " ██╔════╝██╔══██╗██║     ████╗ ████║██╔══██╗"
echo " ██║     ███████║██║     ██╔████╔██║███████║"
echo " ██║     ██╔══██║██║     ██║╚██╔╝██║██╔══██║"
echo " ╚██████╗██║  ██║███████╗██║ ╚═╝ ██║██║  ██║"
echo "  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝"
echo ""
echo " Mental Health Support Platform with Cultural AI"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ Error: Must run this script from the calma-backend root directory${NC}"
    echo -e "${YELLOW}💡 Run: cd calma-backend && ./start-calma.sh${NC}"
    exit 1
fi

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}🔄 Shutting down services...${NC}"

    # Kill AI service if we started it
    if [ ! -z "$AI_SERVICE_PID" ]; then
        echo -e "${YELLOW}🤖 Stopping AI service (PID: $AI_SERVICE_PID)...${NC}"
        kill $AI_SERVICE_PID 2>/dev/null || true
    fi

    # Kill NestJS service if we started it
    if [ ! -z "$NESTJS_PID" ]; then
        echo -e "${YELLOW}🏗️  Stopping NestJS backend (PID: $NESTJS_PID)...${NC}"
        kill $NESTJS_PID 2>/dev/null || true
    fi

    # Kill any remaining processes on our ports
    pkill -f "uvicorn.*app.main" 2>/dev/null || true
    pkill -f "nest.*start" 2>/dev/null || true

    echo -e "${GREEN}✅ Cleanup complete${NC}"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

# Configuration
AI_SERVICE_PORT=8000
NESTJS_PORT=3000
MAX_WAIT_AI=120
MAX_WAIT_NESTJS=30

echo -e "${BLUE}🚀 Starting Calma Mental Health Platform...${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}1️⃣ Checking prerequisites...${NC}"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    exit 1
fi

# Check for Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo -e "${GREEN}✅ Node.js $(node --version) found${NC}"
echo -e "${GREEN}✅ Python $($PYTHON_CMD --version 2>&1 | cut -d' ' -f2) found${NC}"

# Check if npm dependencies are installed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing npm dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}✅ Prerequisites checked${NC}"
echo ""

# Step 2: Start AI Service
echo -e "${BLUE}2️⃣ Starting FastAPI AI Service...${NC}"

# Check if AI service is already running
if curl -s http://localhost:$AI_SERVICE_PORT/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ AI service already running on port $AI_SERVICE_PORT${NC}"
else
    # Check if AI service directory exists
    if [ ! -d "calma-ai-service" ]; then
        echo -e "${RED}❌ AI service directory not found: calma-ai-service${NC}"
        echo -e "${YELLOW}💡 Please ensure the calma-ai-service directory exists${NC}"
        exit 1
    fi

    echo -e "${YELLOW}🤖 Starting AI inference service...${NC}"
    echo -e "${YELLOW}📝 This may take 30-60 seconds to load the model...${NC}"

    # Start AI service in background
    cd calma-ai-service

    # Ensure .env exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${YELLOW}⚠️  Created .env from template. Please review MODEL_PATH setting.${NC}"
        fi
    fi

    # Start the AI service
    $PYTHON_CMD -m app.main > /tmp/calma-ai-service.log 2>&1 &
    AI_SERVICE_PID=$!
    cd ..

    # Wait for AI service to be ready
    echo -e "${YELLOW}⏳ Waiting for AI service to load model...${NC}"
    WAIT_COUNT=0

    while [ $WAIT_COUNT -lt $MAX_WAIT_AI ]; do
        sleep 3
        WAIT_COUNT=$((WAIT_COUNT + 3))

        # Show progress every 15 seconds
        if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
            echo -e "${YELLOW}⏳ Still loading model... (${WAIT_COUNT}s/${MAX_WAIT_AI}s)${NC}"
        fi

        # Check if process died
        if ! kill -0 $AI_SERVICE_PID 2>/dev/null; then
            echo -e "${RED}❌ AI service process died. Check logs:${NC}"
            tail -n 10 /tmp/calma-ai-service.log
            exit 1
        fi

        # Check if service is ready
        if curl -s http://localhost:$AI_SERVICE_PORT/health >/dev/null 2>&1; then
            echo -e "${GREEN}✅ AI service is ready!${NC}"
            break
        fi
    done

    # Final check
    if ! curl -s http://localhost:$AI_SERVICE_PORT/health >/dev/null 2>&1; then
        echo -e "${RED}❌ AI service failed to start within ${MAX_WAIT_AI} seconds${NC}"
        echo -e "${RED}🔍 Check logs: tail -f /tmp/calma-ai-service.log${NC}"
        exit 1
    fi
fi

echo ""

# Step 3: Verify Database Connection
echo -e "${BLUE}3️⃣ Checking database connection...${NC}"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Please create one with DATABASE_URL${NC}"
fi

# Check if Prisma schema exists
if [ ! -f "src/database/prisma/schema.prisma" ]; then
    echo -e "${RED}❌ Prisma schema not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Database configuration found${NC}"
echo ""

# Step 4: Start NestJS Backend
echo -e "${BLUE}4️⃣ Starting NestJS Backend...${NC}"

# Check if backend is already running
if curl -s http://localhost:$NESTJS_PORT/api/chat/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ NestJS backend already running on port $NESTJS_PORT${NC}"
else
    echo -e "${YELLOW}🏗️  Starting NestJS development server...${NC}"

    # Start NestJS in background
    npm run start:dev > /tmp/calma-nestjs.log 2>&1 &
    NESTJS_PID=$!

    # Wait for NestJS to be ready
    echo -e "${YELLOW}⏳ Waiting for NestJS to start...${NC}"
    WAIT_COUNT=0

    while [ $WAIT_COUNT -lt $MAX_WAIT_NESTJS ]; do
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))

        # Check if process died
        if ! kill -0 $NESTJS_PID 2>/dev/null; then
            echo -e "${RED}❌ NestJS process died. Check logs:${NC}"
            tail -n 10 /tmp/calma-nestjs.log
            exit 1
        fi

        # Check if service is ready
        if curl -s http://localhost:$NESTJS_PORT >/dev/null 2>&1; then
            echo -e "${GREEN}✅ NestJS backend is ready!${NC}"
            break
        fi
    done

    # Final check
    if ! curl -s http://localhost:$NESTJS_PORT >/dev/null 2>&1; then
        echo -e "${RED}❌ NestJS failed to start within ${MAX_WAIT_NESTJS} seconds${NC}"
        echo -e "${RED}🔍 Check logs: tail -f /tmp/calma-nestjs.log${NC}"
        exit 1
    fi
fi

echo ""

# Step 5: Integration Test
echo -e "${BLUE}5️⃣ Testing integration...${NC}"

# Test AI service health
if curl -s http://localhost:$AI_SERVICE_PORT/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ AI service health check passed${NC}"
else
    echo -e "${RED}❌ AI service health check failed${NC}"
fi

# Test NestJS chat health
if curl -s http://localhost:$NESTJS_PORT/api/chat/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ NestJS chat service integration working${NC}"
else
    echo -e "${YELLOW}⚠️  NestJS chat health endpoint not responding${NC}"
fi

echo ""

# Success Message
echo -e "${GREEN}🎉 Calma Platform Started Successfully!${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}🌟 CALMA MENTAL HEALTH PLATFORM${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}🤖 AI Inference Service:${NC}"
echo -e "   🌐 URL: http://localhost:$AI_SERVICE_PORT"
echo -e "   📚 API Docs: http://localhost:$AI_SERVICE_PORT/docs"
echo -e "   ❤️  Health: http://localhost:$AI_SERVICE_PORT/health"
echo ""
echo -e "${BLUE}🏗️  NestJS Backend API:${NC}"
echo -e "   🌐 URL: http://localhost:$NESTJS_PORT"
echo -e "   💬 Chat Health: http://localhost:$NESTJS_PORT/api/chat/health"
echo -e "   📊 Full API: http://localhost:$NESTJS_PORT/api"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo -e "   🤖 AI Service: tail -f /tmp/calma-ai-service.log"
echo -e "   🏗️  NestJS: tail -f /tmp/calma-nestjs.log"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop all services${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Keep the script running and monitor services
echo -e "${BLUE}🔄 Monitoring services... (Press Ctrl+C to stop)${NC}"

while true; do
    sleep 10

    # Check if AI service is still running
    if [ ! -z "$AI_SERVICE_PID" ] && ! kill -0 $AI_SERVICE_PID 2>/dev/null; then
        echo -e "${RED}⚠️  AI service process died${NC}"
        break
    fi

    # Check if NestJS is still running
    if [ ! -z "$NESTJS_PID" ] && ! kill -0 $NESTJS_PID 2>/dev/null; then
        echo -e "${RED}⚠️  NestJS process died${NC}"
        break
    fi

    # Periodically check service health
    if ! curl -s http://localhost:$AI_SERVICE_PORT/health >/dev/null 2>&1; then
        echo -e "${RED}⚠️  AI service health check failed${NC}"
    fi

    if ! curl -s http://localhost:$NESTJS_PORT >/dev/null 2>&1; then
        echo -e "${RED}⚠️  NestJS service health check failed${NC}"
    fi
done