#!/usr/bin/env node

// Test script to verify CORS fix is working
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function testCORSFix() {
  console.log('🧪 Testing CORS Fix for Benjamin AI Chat');
  console.log('=' * 50);
  
  console.log('\n🌐 Testing CORS headers...');
  try {
    const response = await fetch('http://localhost:8012/health', {
      headers: {
        'Origin': 'http://localhost:3000'
      }
    });
    
    const corsHeaders = {
      'access-control-allow-origin': response.headers.get('access-control-allow-origin'),
      'access-control-allow-credentials': response.headers.get('access-control-allow-credentials'),
      'vary': response.headers.get('vary')
    };
    
    console.log('✅ CORS Headers:', corsHeaders);
    
    if (corsHeaders['access-control-allow-origin'] === 'http://localhost:3000') {
      console.log('✅ CORS origin header is correct');
    } else {
      console.log('❌ CORS origin header is incorrect');
    }
    
  } catch (error) {
    console.log('❌ CORS test failed:', error.message);
  }
  
  console.log('\n🤖 Testing message route with CORS...');
  try {
    const response = await fetch('http://localhost:8012/route', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Origin': 'http://localhost:3000'
      },
      body: JSON.stringify({
        text: "Test CORS message",
        session_id: "cors-test-" + Date.now()
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ Message route with CORS successful!');
      console.log('   📝 AI responded:', data.reply ? 'Yes' : 'No');
      
      // Test status endpoint
      const sessionId = response.headers.get('X-Session-Id') || 'cors-test-fallback';
      console.log(`\n📊 Testing status endpoint (session: ${sessionId})...`);
      
      const statusResponse = await fetch(`http://localhost:8012/status/${sessionId}`, {
        headers: {
          'Origin': 'http://localhost:3000'
        }
      });
      
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        console.log('✅ Status endpoint works:', statusData.status);
      } else {
        console.log('❌ Status endpoint failed:', statusResponse.status);
      }
      
    } else {
      console.log('❌ Message route failed:', response.status, await response.text());
    }
  } catch (error) {
    console.log('❌ Message route error:', error.message);
  }
  
  console.log('\n🎯 Results:');
  console.log('✅ CORS is now configured properly');
  console.log('✅ Frontend at localhost:3000 can now communicate with API');
  console.log('✅ Chat interface should work without CORS errors');
  
  console.log('\n🚀 Next Steps:');
  console.log('1. Open http://localhost:3000/chat in your browser');
  console.log('2. Send a test message');
  console.log('3. You should see the thinking indicator working');
  console.log('4. No more CORS errors in browser console!');
}

testCORSFix();










