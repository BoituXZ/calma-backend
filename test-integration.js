/**
 * Simple test script to verify NestJS + FastAPI integration
 * Run this after starting both services
 */

const axios = require('axios');

const NESTJS_URL = 'http://localhost:3000/api';
const FASTAPI_URL = 'http://localhost:8000';

async function testIntegration() {
  console.log('🧪 Testing Calma AI Integration...\n');

  try {
    // Test 1: Check FastAPI health
    console.log('1️⃣ Testing FastAPI health...');
    try {
      const fastApiHealth = await axios.get(`${FASTAPI_URL}/health`, { timeout: 5000 });
      console.log('✅ FastAPI Status:', fastApiHealth.data.status);
      console.log('📊 Model Status:', fastApiHealth.data.model_status);
    } catch (error) {
      console.log('❌ FastAPI not available:', error.message);
      console.log('💡 Make sure to start the FastAPI service first:');
      console.log('   cd calma-ai-service && uvicorn app.main:app --reload');
      return;
    }

    // Test 2: Check NestJS chat health
    console.log('\n2️⃣ Testing NestJS chat service health...');
    try {
      const nestjsHealth = await axios.get(`${NESTJS_URL}/chat/health`, { timeout: 5000 });
      console.log('✅ NestJS Chat Status:', nestjsHealth.data.status);
      console.log('🤖 AI Service:', nestjsHealth.data.ai_service);
    } catch (error) {
      console.log('❌ NestJS chat service not available:', error.message);
      console.log('💡 Make sure to start the NestJS service:');
      console.log('   npm run start:dev');
      return;
    }

    // Test 3: Test direct FastAPI inference
    console.log('\n3️⃣ Testing direct FastAPI inference...');
    try {
      const inferenceRequest = {
        message: "I'm feeling stressed about my studies",
        parameters: {
          temperature: 0.8,
          max_tokens: 150
        }
      };

      const inferenceResponse = await axios.post(
        `${FASTAPI_URL}/infer`,
        inferenceRequest,
        { timeout: 30000 }
      );

      console.log('✅ AI Response generated successfully');
      console.log('📝 Response length:', inferenceResponse.data.response.length, 'characters');
      console.log('😊 Mood detected:', inferenceResponse.data.metadata.mood_detected);
      console.log('🎯 Confidence:', inferenceResponse.data.metadata.confidence);
      console.log('📚 Suggested resources:', inferenceResponse.data.suggested_resources);
      console.log('🌍 Cultural elements:', inferenceResponse.data.cultural_elements_detected);
    } catch (error) {
      console.log('❌ FastAPI inference failed:', error.response?.data || error.message);
      return;
    }

    // Test 4: Test end-to-end integration (requires a real user ID and database)
    console.log('\n4️⃣ Testing end-to-end integration...');
    console.log('⚠️  This test requires a valid user ID in your database.');
    console.log('💡 You can test this manually using:');
    console.log('   POST http://localhost:3000/api/chat/message');
    console.log('   Body: { "userId": "your-user-uuid", "message": "Hello, I need help" }');

    console.log('\n🎉 Integration tests completed successfully!');
    console.log('\n📋 Next steps:');
    console.log('   1. Create a test user in your database');
    console.log('   2. Test the full chat flow with a real frontend');
    console.log('   3. Monitor logs for any integration issues');

  } catch (error) {
    console.error('❌ Integration test failed:', error.message);
  }
}

// Helper function to check if services are running
async function checkServices() {
  console.log('🔍 Checking service availability...\n');

  const services = [
    { name: 'FastAPI AI Service', url: `${FASTAPI_URL}/health` },
    { name: 'NestJS Backend', url: `${NESTJS_URL}/chat/health` },
  ];

  for (const service of services) {
    try {
      await axios.get(service.url, { timeout: 3000 });
      console.log(`✅ ${service.name} is running`);
    } catch (error) {
      console.log(`❌ ${service.name} is not available`);
    }
  }

  console.log('\n');
}

// Run the tests
if (require.main === module) {
  checkServices().then(() => testIntegration());
}

module.exports = { testIntegration, checkServices };