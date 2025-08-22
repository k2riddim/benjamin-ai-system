#!/usr/bin/env python3
"""
Test script to demonstrate the thinking status API functionality
"""

import requests
import time
import json
from typing import Dict, Any

# Configuration
AGENTIC_API_BASE = "http://localhost:8012"
TEST_SESSION_ID = "test-session-123"

def test_thinking_status():
    """Test the thinking status API functionality"""
    print("🧪 Testing Benjamin AI Thinking Status API")
    print("=" * 50)
    
    # Test 1: Check initial status (should be idle)
    print("\n1️⃣ Testing initial status...")
    try:
        response = requests.get(f"{AGENTIC_API_BASE}/status/{TEST_SESSION_ID}")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Initial status: {status}")
            assert status["status"] == "idle", f"Expected idle, got {status['status']}"
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return False
    
    # Test 2: Send a simple route request and monitor status
    print("\n2️⃣ Testing route request with status tracking...")
    try:
        # Start the request in a separate thread to monitor status
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def make_route_request():
            try:
                response = requests.post(
                    f"{AGENTIC_API_BASE}/route",
                    json={
                        "text": "What's my current training load?",
                        "session_id": TEST_SESSION_ID
                    },
                    timeout=30
                )
                result_queue.put(response)
            except Exception as e:
                result_queue.put(e)
        
        # Start the request
        request_thread = threading.Thread(target=make_route_request)
        request_thread.start()
        
        # Monitor status while request is processing
        status_updates = []
        max_checks = 20
        check_count = 0
        
        while request_thread.is_alive() and check_count < max_checks:
            try:
                status_response = requests.get(f"{AGENTIC_API_BASE}/status/{TEST_SESSION_ID}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    status_updates.append(status)
                    print(f"   📊 Status: {status['status']} - {status['details']}")
                    if status.get('agents'):
                        print(f"      🤖 Active agents: {', '.join(status['agents'])}")
            except Exception as e:
                print(f"   ⚠️ Error checking status: {e}")
            
            time.sleep(0.5)
            check_count += 1
        
        # Wait for request to complete
        request_thread.join(timeout=10)
        
        # Get final result
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, requests.Response):
                if result.status_code == 200:
                    print(f"✅ Route request completed successfully")
                    route_result = result.json()
                    print(f"   📝 Response preview: {route_result.get('reply', '')[:100]}...")
                else:
                    print(f"❌ Route request failed: {result.status_code}")
                    return False
            else:
                print(f"❌ Route request error: {result}")
                return False
        
        print(f"\n   📈 Captured {len(status_updates)} status updates:")
        for i, status in enumerate(status_updates):
            print(f"      {i+1}. {status['status']}: {status['details']}")
            
    except Exception as e:
        print(f"❌ Error during route test: {e}")
        return False
    
    # Test 3: Test daily discussion with status tracking
    print("\n3️⃣ Testing daily discussion with status tracking...")
    try:
        response = requests.post(f"{AGENTIC_API_BASE}/daily-discussion", json={})
        if response.status_code == 200:
            daily_result = response.json()
            session_id = response.headers.get('X-Session-Id')
            print(f"✅ Daily discussion completed")
            print(f"   🆔 Session ID: {session_id}")
            
            if session_id:
                # Check final status
                status_response = requests.get(f"{AGENTIC_API_BASE}/status/{session_id}")
                if status_response.status_code == 200:
                    final_status = status_response.json()
                    print(f"   📊 Final status: {final_status}")
        else:
            print(f"❌ Daily discussion failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error during daily discussion test: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("\n📖 How to use the thinking status API:")
    print("   1. Send a request to /route or /daily-discussion")
    print("   2. Extract session_id from request headers or use your own")
    print("   3. Poll GET /status/{session_id} to get real-time thinking status")
    print("   4. Status values: 'idle', 'thinking', 'complete', 'error'")
    print("   5. Details field provides human-readable status description")
    print("   6. Agents field shows which AI specialists are currently active")
    
    return True

if __name__ == "__main__":
    success = test_thinking_status()
    exit(0 if success else 1)
