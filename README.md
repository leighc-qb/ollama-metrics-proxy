> ### Fork notice
>
> Fork of [`elliotfehr/ollama-metrics-proxy`](https://github.com/elliotfehr/ollama-metrics-proxy).
>
> It adds a **null-content filter** that repairs the malformed
> `{"role":"assistant","content":null}` messages GitHub Copilot sends, which Ollama
> otherwise rejects with `400 invalid message content type: <nil>`, plus a small
> upstream-URL fix. All Prometheus metrics behave exactly as upstream.
>
> See [Null-content filter (fork addition)](#null-content-filter-fork-addition).
>
> `main` tracks upstream exactly; changes live on the
> [`null-content-filter-fix`](https://github.com/leighc-qb/ollama-metrics-proxy/tree/null-content-filter-fix) branch.

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


---

## Null-content filter (fork addition)

*Documents functionality that exists only in this fork.*

### The problem

GitHub Copilot (CLI and desktop app) sometimes emits an assistant message whose
`content` is JSON `null` and which carries **no** `tool_calls`:

```json
{"role": "assistant", "content": null}
```

The OpenAI API tolerates this. Ollama does not:

```
400 invalid message content type: <nil>
```

Because the bad message stays in the conversation history it is replayed on every
later turn, so a session fails permanently once one appears.

Upstream bug reports:

- GitHub Copilot CLI — <https://github.com/github/copilot-cli/issues/4269>
- GitHub Copilot app — <https://github.com/github/app/issues/2140>

### The fix

`repairEmptyAssistantMessages()` in [`proxy.go`](proxy.go) rewrites `content: null`
to `content: ""` before the request reaches Ollama.

It runs only when **all** of these hold:

- the method is `POST`
- the `Content-Type` is JSON (absent, `application/json`, or `*+json`)
- the endpoint is `/v1/chat/completions` or `/api/chat`

### Repair predicate

Within such a request, a message is rewritten **only** when:

| Condition | Required value |
|---|---|
| `role` | exactly `"assistant"` |
| `content` | key **present** and JSON `null` |
| `tool_calls` | absent, `null`, or `[]` |
| `function_call` | absent or `null` |

An assistant message with **real** `tool_calls` is a *legitimate* use of
`content: null` and is never rewritten -- doing so would break tool calling.

### Why it operates on `json.RawMessage`

The filter decodes into `map[string]json.RawMessage`, not `map[string]any`, and
rewrites only the `content` value of the offending messages. Every other field keeps
its **exact original bytes**.

This matters. A naive `map[string]any` implementation round-trips every JSON number
through `float64` and silently corrupts large integers:

```
seed 9007199254740993  ->  9007199254740992     # off by one, beyond 2^53
seed 12345678901234567890 -> 12345678901234567000
```

Options like `seed` and `num_ctx` are passed straight through to Ollama, so that
corruption would be a real, hard-to-diagnose bug. It is locked out by
`TestRepairPreservesLargeNumbersExactly` in
[`nullcontent_regression_test.go`](nullcontent_regression_test.go).

When no message needs repair the body is forwarded **byte-for-byte unchanged**, so
the filter is a no-op for every non-Copilot client.

### Example

Before (rejected by Ollama):

```json
{"messages": [
  {"role": "user", "content": "hi"},
  {"role": "assistant", "content": null}
]}
```

After (accepted):

```json
{"messages": [
  {"role": "user", "content": "hi"},
  {"role": "assistant", "content": ""}
]}
```

Left alone -- this one has real `tool_calls`:

```json
{"role": "assistant", "content": null,
 "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls"}}]}
```

### Observability

Each repaired request logs one line:

```
copilot_null_content_repaired path=/v1/chat/completions count=1
```

`count` is the number of messages repaired. Nothing is logged when nothing needed
repair.

### Upstream URL handling

Also fixed here: the proxy now preserves the query string and any base path when
building the upstream request, instead of concatenating the endpoint onto the
configured URL.

### Tests

```sh
go test -run Repair -v ./...
```

- [`proxy_test.go`](proxy_test.go) -- predicate table, endpoint coverage, and the
  non-JSON content-type skip
- [`nullcontent_regression_test.go`](nullcontent_regression_test.go) -- numeric
  precision, unrelated-field preservation, and byte-identical no-op behaviour
