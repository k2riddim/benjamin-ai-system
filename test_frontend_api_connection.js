#!/usr/bin/env node

// Test script to verify frontend API connections work
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function testAPIs() {
  console.log('🧪 Testing Frontend API Connections');
  console.log('=' * 50);
  
  // Test Data API
  console.log('\n📊 Testing Data API (localhost:8010)...');
  try {
    const response = await fetch('http://localhost:8010/health');
    const data = await response.json();
    console.log('✅ Data API:', data);
  } catch (error) {
    console.log('❌ Data API Error:', error.message);
  }
  
  // Test Agentic API
  console.log('\n🤖 Testing Agentic API (localhost:8012)...');
  try {
    const response = await fetch('http://localhost:8012/health');
    const data = await response.json();
    console.log('✅ Agentic API:', data);
  } catch (error) {
    console.log('❌ Agentic API Error:', error.message);
  }
  
  // Test a simple message route
  console.log('\n💬 Testing Message Route...');
  try {
    const response = await fetch('http://localhost:8012/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: "Hello, test message",
        session_id: "test-frontend-123"
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ Message Route successful');
      console.log('   📝 Response preview:', data.reply?.substring(0, 100) + '...');
      
      // Test status endpoint
      const sessionId = response.headers.get('X-Session-Id') || 'test-frontend-123';
      console.log(`\n📊 Testing Status Endpoint (session: ${sessionId})...`);
      
      const statusResponse = await fetch(`http://localhost:8012/status/${sessionId}`);
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        console.log('✅ Status API:', statusData);
      } else {
        console.log('❌ Status API failed:', statusResponse.status);
      }
    } else {
      console.log('❌ Message Route failed:', response.status, await response.text());
    }
  } catch (error) {
    console.log('❌ Message Route Error:', error.message);
  }
  
  // Test Frontend
  console.log('\n🌐 Testing Frontend (localhost:3000)...');
  try {
    const response = await fetch('http://localhost:3000');
    if (response.ok) {
      console.log('✅ Frontend is accessible');
      console.log('   🔗 Chat page: http://localhost:3000/chat');
    } else {
      console.log('❌ Frontend failed:', response.status);
    }
  } catch (error) {
    console.log('❌ Frontend Error:', error.message);
  }
  
  console.log('\n🎯 Next Steps:');
  console.log('1. Open http://localhost:3000/chat in your browser');
  console.log('2. Try sending a message like "What\'s my training status?"');
  console.log('3. Watch for the thinking indicator with real-time status updates');
  console.log('4. Check browser console (F12) for any JavaScript errors');
  
  console.log('\n🐛 If chat is still blank:');
  console.log('1. Open browser Developer Tools (F12)');
  console.log('2. Check Console tab for errors');
  console.log('3. Check Network tab to see if API calls are failing');
  console.log('4. Try hard refresh (Ctrl+F5) to clear cache');
}

testAPIs();





















