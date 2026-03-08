# Teler Gemini Bridge

A reference integration between Teler and Gemini Live API, based on Media Streaming over WebSockets.

## Setup

1. **Clone and configure:**
   ```bash
   git clone https://github.com/frejun-tech/teler-gemini-bridge.git
   cd teler-gemini-bridge
   cp env.example .env
   # Edit .env with your actual values
   ```

2. **Run with Docker:**
   ```bash
   docker compose up -d --build
   ```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Your Google API key |
| `GEMINI_MODEL` | Gemini model to use |
| `GEMINI_SYSTEM_MESSAGE` | System prompt for Gemini |
| `GEMINI_INITIAL_PROMPT` | Initial prompt spoken at call start |
| `GEMINI_AUDIO_CHUNK_COUNT` | Audio chunks to buffer before sending |
| `SERVER_DOMAIN` | Your public server domain |
| `TELER_API_KEY` | Your Teler API key |

## API Endpoints

- `POST /api/v1/calls/initiate-call` - Start a new call
- `POST /api/v1/calls/flow` - Get call flow configuration
- `WebSocket /api/v1/calls/media-stream` - Audio streaming
- `POST /api/v1/webhooks/receiver` - Teler webhook receiver
- `GET /health` - Service status

### Call Initiation Example

```bash
curl -X POST "http://localhost:8000/api/v1/calls/initiate-call" \
  -H "Content-Type: application/json" \
  -d '{
    "from_number": "+1234567890",
    "to_number": "+0987654321"
  }'
```
