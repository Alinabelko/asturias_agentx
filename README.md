# AgentX Research — tau2-bench Purple Agent

Customer service agent for [tau2-bench](https://github.com/RDI-Foundation/tau2-agentbeats) evaluation on [AgentBeats](https://agentbeats.dev).

## Stack

- A2A protocol (a2a-sdk)
- OpenAI API (gpt-5.4-mini)
- Python 3.11+

## Run locally

```bash
cp .env.example .env
# add OPENAI_API_KEY to .env

python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt   # or: pip install "a2a-sdk[http-server]" openai python-dotenv uvicorn

python src/server.py --port 9019
```

Agent card: `http://localhost:9019/.well-known/agent.json`

## Deploy

```bash
docker build --platform linux/amd64 -t alinabelko/agentx-research:latest .
docker push alinabelko/agentx-research:latest
```

## Results

**42% task pass rate** on tau2-bench airline domain leaderboard.

## Agent

- Receives customer service tasks via A2A messages
- Responds with JSON: `{"name": "tool_or_respond", "arguments": {...}}`
- Maintains conversation history per context
- Domains: airline, retail, telecom
