# ollama-metrics-proxy

A lightweight reverse proxy that sits in front of [Ollama](https://ollama.com) and exposes Prometheus metrics for inference requests. It transparently captures token counts, request durations, and generation speed without requiring any changes to your Ollama setup or client applications.

## Features

- Transparent reverse proxy — clients connect to the proxy instead of Ollama directly
- Supports all three API formats Ollama exposes:
  - **Ollama native** (`/api/generate`, `/api/chat`) — full metrics including tokens/sec, eval durations, model load time
  - **OpenAI-compatible** (`/v1/chat/completions`) — token counts from usage fields
  - **Anthropic-compatible** (`/v1/messages`) — token counts from streaming events
- Handles long-lived streaming requests with no timeouts
- Non-inference endpoints (model management, health checks, etc.) are passed through unchanged

## Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `ollama_requests_total` | counter | `model`, `endpoint` | Completed inference requests |
| `ollama_prompt_tokens_total` | counter | `model` | Prompt/input tokens processed |
| `ollama_completion_tokens_total` | counter | `model` | Completion/output tokens generated |
| `ollama_request_duration_seconds` | histogram | `model`, `endpoint` | End-to-end request duration |
| `ollama_active_requests` | gauge | `model`, `endpoint` | Currently in-flight requests |
| `ollama_tokens_per_second` | gauge | `model` | Most recent generation speed* |
| `ollama_prompt_eval_seconds_total` | counter | `model` | Time evaluating prompts* |
| `ollama_token_generation_seconds_total` | counter | `model` | Time generating tokens* |
| `ollama_model_load_seconds_total` | counter | `model` | Time loading models* |

*Available only through Ollama native endpoints (`/api/generate`, `/api/chat`).

## Install

### From source

```bash
go install github.com/elliotfehr/ollama-metrics-proxy@latest
```

### Build locally

```bash
git clone https://github.com/elliotfehr/ollama-metrics-proxy.git
cd ollama-metrics-proxy
go build -o ollama-metrics-proxy .
```

## Usage

```bash
ollama-metrics-proxy \
  --listen :11435 \
  --metrics-listen :9836 \
  --ollama-url http://localhost:11434
```

Then point your clients at `http://localhost:11435` instead of `http://localhost:11434`. Scrape metrics from `http://localhost:9836/metrics`.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--listen` | `:11435` | Address the proxy listens on |
| `--metrics-listen` | `:9836` | Address for the Prometheus `/metrics` endpoint |
| `--ollama-url` | `http://localhost:11434` | Ollama backend URL |

### Example: Claude Code with a local Ollama model

```bash
ANTHROPIC_BASE_URL="http://localhost:11435" claude
```

### Example: Prometheus scrape config

```yaml
scrape_configs:
  - job_name: ollama
    static_configs:
      - targets: ["localhost:9836"]
```

### Example: systemd service

```ini
[Unit]
Description=Ollama Metrics Proxy
After=network.target ollama.service

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStart=/usr/local/bin/ollama-metrics-proxy \
  --listen=:11435 \
  --metrics-listen=:9836 \
  --ollama-url=http://localhost:11434

[Install]
WantedBy=multi-user.target
```

## How it works

The proxy intercepts requests to inference endpoints and streams the response back to the client line-by-line. As each chunk passes through, it inspects the data for token usage information:

- **Ollama native**: The final JSON chunk (where `done: true`) contains `prompt_eval_count`, `eval_count`, and timing fields.
- **OpenAI-compatible**: The proxy injects `stream_options: {"include_usage": true}` into streaming requests so the server returns a `usage` object in the final SSE chunk.
- **Anthropic-compatible**: Token counts are extracted from `message_start` (input tokens) and `message_delta` (output tokens) SSE events.

All other endpoints (`/api/tags`, `/api/show`, `/api/ps`, health checks, etc.) are forwarded without inspection.

## License

[MIT](LICENSE)
