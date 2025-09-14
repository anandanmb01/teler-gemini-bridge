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

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with Docker:**
   ```bash
   docker compose up -d --build
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google API key | Required |
| `GEMINI_MODEL` | Gemini model to use | gemini-2.5-flash-preview-native-audio-dialog |
| `GEMINI_SYSTEM_MESSAGE` | System prompt for Gemini | You are a friendly and helpful AI voice assistant... |
| `GEMINI_AUDIO_CHUNK_COUNT` | Audio chunks to buffer before sending | 5 |
| `SERVER_DOMAIN` | Your ngrok domain (auto-detected) | Required |
| `TELER_API_KEY` | Your Teler API key | Required |
| `NGROK_AUTHTOKEN` | Your ngrok auth token | Required |

## API Endpoints

- `GET /` - Health check with server domain and provider status
- `GET /health` - Service status
- `GET /ngrok-status` - Current ngrok status and URL
- `POST /api/v1/calls/initiate-call` - Start a new call
- `POST /api/v1/calls/flow` - Get call flow configuration
- `WebSocket /api/v1/calls/media-stream` - Audio streaming
- `POST /api/v1/webhooks/receiver` - Teler webhook receiver

### Call Initiation Example

```bash
curl -X POST "http://localhost:8000/api/v1/calls/initiate-call" \
  -H "Content-Type: application/json" \
  -d '{
    "from_number": "+1234567890",
    "to_number": "+0987654321"
  }'
```

## Features

- **Gemini Live API Integration** - Direct integration with Google's Gemini Live API
- **Dynamic ngrok URL detection** - Automatically detects current ngrok domain
- **Audio buffering** - Buffers audio chunks before sending to Gemini for optimal performance
- **Sample rate conversion** - High-quality conversion between Gemini (24kHz output) and Teler (8kHz required for playback)
- **Real-time streaming** - WebSocket-based audio streaming between Teler and Gemini
- **High-quality audio processing** - Uses scipy for professional-grade audio resampling
