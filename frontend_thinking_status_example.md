# Frontend Thinking Status Integration

This document shows how a frontend application can integrate with the Benjamin AI thinking status API to provide real-time feedback to users.

## API Overview

The thinking status system provides these endpoints:

- `POST /route` - Process user messages (returns session ID in headers)
- `POST /daily-discussion` - Generate daily workout (returns session ID in headers)  
- `GET /status/{session_id}` - Get current thinking status for a session

## Status States

| Status | Description | UI Suggestion |
|--------|-------------|---------------|
| `idle` | No processing | Hidden or "Ready" |
| `thinking` | AI agents working | Spinner + status message |
| `complete` | Processing done | Hide spinner, show result |
| `error` | Processing failed | Show error message |

## Frontend Implementation Example

### JavaScript/React Example

```javascript
class BenjaminAIChat {
  constructor() {
    this.apiBase = 'http://localhost:8012';
    this.currentSessionId = null;
    this.statusPoller = null;
  }

  async sendMessage(message) {
    // Show initial thinking state
    this.updateThinkingStatus('thinking', 'Processing your request...');
    
    try {
      // Send message to agentic API
      const response = await fetch(`${this.apiBase}/route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: message,
          session_id: this.generateSessionId()
        })
      });
      
      // Extract session ID from headers
      this.currentSessionId = response.headers.get('X-Session-Id') || this.currentSessionId;
      
      // Start polling for status updates
      this.startStatusPolling();
      
      // Wait for completion
      const result = await response.json();
      
      // Stop polling and show result
      this.stopStatusPolling();
      this.showResponse(result.reply);
      
    } catch (error) {
      this.stopStatusPolling();
      this.showError('Failed to send message');
    }
  }

  async requestDailyWorkout() {
    this.updateThinkingStatus('thinking', 'Preparing your daily workout...');
    
    try {
      const response = await fetch(`${this.apiBase}/daily-discussion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      this.currentSessionId = response.headers.get('X-Session-Id');
      this.startStatusPolling();
      
      const result = await response.json();
      this.stopStatusPolling();
      this.showResponse(result.telegram_message);
      
    } catch (error) {
      this.stopStatusPolling();
      this.showError('Failed to generate workout');
    }
  }

  startStatusPolling() {
    if (!this.currentSessionId) return;
    
    this.statusPoller = setInterval(async () => {
      try {
        const response = await fetch(`${this.apiBase}/status/${this.currentSessionId}`);
        const status = await response.json();
        
        this.updateThinkingStatus(status.status, status.details, status.agents);
        
        // Stop polling when complete or error
        if (status.status === 'complete' || status.status === 'error') {
          this.stopStatusPolling();
        }
      } catch (error) {
        console.warn('Status polling failed:', error);
      }
    }, 500); // Poll every 500ms
  }

  stopStatusPolling() {
    if (this.statusPoller) {
      clearInterval(this.statusPoller);
      this.statusPoller = null;
    }
  }

  updateThinkingStatus(status, details, agents = []) {
    const statusElement = document.getElementById('thinking-status');
    
    if (status === 'idle' || status === 'complete') {
      statusElement.style.display = 'none';
      return;
    }
    
    statusElement.style.display = 'block';
    
    if (status === 'thinking') {
      statusElement.innerHTML = `
        <div class="thinking-indicator">
          <div class="spinner"></div>
          <div class="status-text">${details}</div>
          ${agents.length > 0 ? `<div class="agents">Consulting: ${agents.join(', ')}</div>` : ''}
        </div>
      `;
    } else if (status === 'error') {
      statusElement.innerHTML = `
        <div class="error-indicator">
          <span class="error-icon">⚠️</span>
          <span class="error-text">${details}</span>
        </div>
      `;
    }
  }

  generateSessionId() {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  showResponse(message) {
    const messagesContainer = document.getElementById('messages');
    messagesContainer.innerHTML += `
      <div class="ai-message">
        <div class="message-content">${message}</div>
        <div class="message-time">${new Date().toLocaleTimeString()}</div>
      </div>
    `;
    this.updateThinkingStatus('idle');
  }

  showError(error) {
    const messagesContainer = document.getElementById('messages');
    messagesContainer.innerHTML += `
      <div class="error-message">
        <div class="message-content">❌ ${error}</div>
      </div>
    `;
    this.updateThinkingStatus('idle');
  }
}
```

### CSS for Thinking Indicator

```css
.thinking-indicator {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  background: #f0f8ff;
  border: 1px solid #e1ecf4;
  border-radius: 8px;
  margin: 8px 0;
}

.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e1ecf4;
  border-top: 2px solid #0066cc;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 12px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.status-text {
  font-weight: 500;
  color: #0066cc;
  flex-grow: 1;
}

.agents {
  font-size: 0.9em;
  color: #666;
  margin-left: 12px;
}

.error-indicator {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  background: #fff5f5;
  border: 1px solid #fed7d7;
  border-radius: 8px;
  margin: 8px 0;
}

.error-icon {
  margin-right: 8px;
}

.error-text {
  color: #c53030;
  font-weight: 500;
}
```

### HTML Structure

```html
<div id="chat-container">
  <div id="messages"></div>
  <div id="thinking-status" style="display: none;"></div>
  
  <div class="input-area">
    <input type="text" id="message-input" placeholder="Type your message...">
    <button onclick="chat.sendMessage(document.getElementById('message-input').value)">Send</button>
    <button onclick="chat.requestDailyWorkout()">Daily Workout</button>
  </div>
</div>

<script>
const chat = new BenjaminAIChat();
</script>
```

## Status Flow Examples

### Message Flow
1. User sends "What's my VO2 max?"
2. Status: `thinking` - "Processing your request..."
3. Status: `thinking` - "Understanding your request..."  
4. Status: `thinking` - "Consulting data_analyst..."
5. Status: `complete` - "Response ready"
6. Show final response

### Daily Workout Flow
1. User clicks "Daily Workout"
2. Status: `thinking` - "Preparing daily workout recommendation..."
3. Status: `thinking` - "Generating workout plan..."
4. Status: `thinking` - "Consulting running_coach, nutritionist, psychologist..."
5. Status: `complete` - "Daily workout ready"
6. Show workout plan

## Benefits

- **User Engagement**: Users see the AI is actively working
- **Transparency**: Users understand which specialists are involved
- **Better UX**: No "dead time" during long AI processing
- **Progress Indication**: Clear feedback on processing stages
- **Error Handling**: Immediate feedback when something goes wrong

## Implementation Tips

1. **Polling Frequency**: 500ms provides good responsiveness without overwhelming the server
2. **Fallback Handling**: Always have a timeout in case status polling fails
3. **Visual Feedback**: Use spinners, progress bars, or typing indicators
4. **Agent Display**: Show which AI specialists are working (optional)
5. **Error Recovery**: Provide clear error messages and retry options
