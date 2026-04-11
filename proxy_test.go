package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// testProxy creates a proxy backed by the given handler and a fresh
// Prometheus registry, returning the proxy and the registry for assertions.
func testProxy(t *testing.T, backend http.Handler) (*proxy, *prometheus.Registry) {
	t.Helper()
	srv := httptest.NewServer(backend)
	t.Cleanup(srv.Close)

	target, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	reg := prometheus.NewRegistry()
	p := newProxy(target, reg)
	return p, reg
}

func postJSON(t *testing.T, handler http.Handler, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	return rr
}

func counterValue(reg *prometheus.Registry, name string, labels prometheus.Labels) float64 {
	families, _ := reg.Gather()
	for _, f := range families {
		if f.GetName() != name {
			continue
		}
		for _, m := range f.GetMetric() {
			if matchLabels(m, labels) {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func gaugeValue(reg *prometheus.Registry, name string, labels prometheus.Labels) float64 {
	families, _ := reg.Gather()
	for _, f := range families {
		if f.GetName() != name {
			continue
		}
		for _, m := range f.GetMetric() {
			if matchLabels(m, labels) {
				return m.GetGauge().GetValue()
			}
		}
	}
	return 0
}

func histogramCount(reg *prometheus.Registry, name string, labels prometheus.Labels) uint64 {
	families, _ := reg.Gather()
	for _, f := range families {
		if f.GetName() != name {
			continue
		}
		for _, m := range f.GetMetric() {
			if matchLabels(m, labels) {
				return m.GetHistogram().GetSampleCount()
			}
		}
	}
	return 0
}

func matchLabels(m *dto.Metric, labels prometheus.Labels) bool {
	lps := m.GetLabel()
	if len(lps) != len(labels) {
		return false
	}
	for _, lp := range lps {
		v, ok := labels[lp.GetName()]
		if !ok || v != lp.GetValue() {
			return false
		}
	}
	return true
}

// --- Ollama native tests ---

func TestOllamaNonStreaming(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(ollamaStats{
			Model:              "llama3",
			Done:               true,
			PromptEvalCount:    10,
			PromptEvalDuration: 500_000_000,
			EvalCount:          20,
			EvalDuration:       1_000_000_000,
			TotalDuration:      2_000_000_000,
			LoadDuration:       100_000_000,
		})
	})

	p, reg := testProxy(t, backend)
	streamFalse := `{"model":"llama3","prompt":"hi","stream":false}`
	rr := postJSON(t, p, "/api/generate", streamFalse)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "llama3", "endpoint": "/api/generate"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 10, prometheus.Labels{"model": "llama3"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 20, prometheus.Labels{"model": "llama3"})
	assertCounter(t, reg, "ollama_prompt_eval_seconds_total", 0.5, prometheus.Labels{"model": "llama3"})
	assertCounter(t, reg, "ollama_token_generation_seconds_total", 1.0, prometheus.Labels{"model": "llama3"})
	assertCounter(t, reg, "ollama_model_load_seconds_total", 0.1, prometheus.Labels{"model": "llama3"})
	assertGauge(t, reg, "ollama_tokens_per_second", 20.0, prometheus.Labels{"model": "llama3"})
}

func TestOllamaStreaming(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chunks := []string{
			`{"model":"llama3","done":false,"response":"hello"}`,
			`{"model":"llama3","done":false,"response":" world"}`,
			`{"model":"llama3","done":true,"response":"","prompt_eval_count":5,"prompt_eval_duration":200000000,"eval_count":10,"eval_duration":500000000,"total_duration":1000000000,"load_duration":50000000}`,
		}
		for _, c := range chunks {
			fmt.Fprintln(w, c)
		}
	})

	p, reg := testProxy(t, backend)
	// Ollama streams by default when stream is omitted
	rr := postJSON(t, p, "/api/chat", `{"model":"llama3","messages":[]}`)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	// Verify response was passed through
	body := rr.Body.String()
	if !strings.Contains(body, "hello") || !strings.Contains(body, "world") {
		t.Errorf("response body missing streamed content: %s", body)
	}

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "llama3", "endpoint": "/api/chat"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 5, prometheus.Labels{"model": "llama3"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 10, prometheus.Labels{"model": "llama3"})
}

// --- OpenAI-compatible tests ---

func TestOpenAINonStreaming(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openAIResponse{
			Model: "gpt-4",
			Usage: &openAIUsage{PromptTokens: 15, CompletionTokens: 25, TotalTokens: 40},
		})
	})

	p, reg := testProxy(t, backend)
	rr := postJSON(t, p, "/v1/chat/completions", `{"model":"gpt-4","messages":[]}`)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "gpt-4", "endpoint": "/v1/chat/completions"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 15, prometheus.Labels{"model": "gpt-4"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 25, prometheus.Labels{"model": "gpt-4"})
}

func TestOpenAIStreaming(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify stream_options was injected
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"include_usage":true`) {
			t.Error("stream_options.include_usage not injected into request")
		}

		lines := []string{
			`data: {"model":"gpt-4","choices":[{"delta":{"content":"hi"}}]}`,
			`data: {"model":"gpt-4","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":12,"total_tokens":20}}`,
			`data: [DONE]`,
		}
		for _, l := range lines {
			fmt.Fprintln(w, l)
		}
	})

	p, reg := testProxy(t, backend)
	rr := postJSON(t, p, "/v1/chat/completions", `{"model":"gpt-4","messages":[],"stream":true}`)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "gpt-4", "endpoint": "/v1/chat/completions"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 8, prometheus.Labels{"model": "gpt-4"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 12, prometheus.Labels{"model": "gpt-4"})
}

// --- Anthropic tests ---

func TestAnthropicNonStreaming(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(anthropicResponse{
			Model: "claude-3",
			Usage: &anthropicUsage{InputTokens: 30, OutputTokens: 50},
		})
	})

	p, reg := testProxy(t, backend)
	rr := postJSON(t, p, "/v1/messages", `{"model":"claude-3","messages":[]}`)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "claude-3", "endpoint": "/v1/messages"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 30, prometheus.Labels{"model": "claude-3"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 50, prometheus.Labels{"model": "claude-3"})
}

func TestAnthropicStreaming(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lines := []string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"model":"claude-3","usage":{"input_tokens":20,"output_tokens":0}}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":35}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
		}
		for _, l := range lines {
			fmt.Fprintln(w, l)
		}
	})

	p, reg := testProxy(t, backend)
	rr := postJSON(t, p, "/v1/messages", `{"model":"claude-3","messages":[],"stream":true}`)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "claude-3", "endpoint": "/v1/messages"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 20, prometheus.Labels{"model": "claude-3"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 35, prometheus.Labels{"model": "claude-3"})
}

// --- Routing tests ---

func TestNonInferenceEndpointPassthrough(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"models":[]}`))
	})

	p, _ := testProxy(t, backend)
	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "models") {
		t.Error("passthrough response missing expected content")
	}
}

func TestUpstreamError(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	})

	p, reg := testProxy(t, backend)
	rr := postJSON(t, p, "/api/generate", `{"model":"llama3","prompt":"hi","stream":false}`)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", rr.Code)
	}
	// No metrics should be recorded for error responses
	if v := counterValue(reg, "ollama_requests_total", prometheus.Labels{"model": "llama3", "endpoint": "/api/generate"}); v != 0 {
		t.Errorf("expected 0 requests recorded on error, got %v", v)
	}
}

func TestModelFallback(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return stats with empty model field
		json.NewEncoder(w).Encode(ollamaStats{
			Done:            true,
			PromptEvalCount: 1,
			EvalCount:       1,
		})
	})

	p, reg := testProxy(t, backend)
	postJSON(t, p, "/api/generate", `{"model":"my-model","prompt":"hi","stream":false}`)

	// Should fall back to request model
	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "my-model", "endpoint": "/api/generate"})
}

func TestUnknownModelFallback(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(ollamaStats{Done: true, EvalCount: 1})
	})

	p, reg := testProxy(t, backend)
	postJSON(t, p, "/api/generate", `{"prompt":"hi","stream":false}`)

	assertCounter(t, reg, "ollama_requests_total", 1, prometheus.Labels{"model": "unknown", "endpoint": "/api/generate"})
}

// --- Helper: injectStreamUsage ---

func TestInjectStreamUsage(t *testing.T) {
	input := `{"model":"m","stream":true,"messages":[]}`
	result := injectStreamUsage([]byte(input))

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(result, &parsed); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	so, ok := parsed["stream_options"]
	if !ok {
		t.Fatal("stream_options not found in result")
	}
	if !strings.Contains(string(so), `"include_usage":true`) {
		t.Errorf("unexpected stream_options: %s", so)
	}
	// Original fields preserved
	if _, ok := parsed["model"]; !ok {
		t.Error("original model field lost")
	}
}

func TestInjectStreamUsageInvalidJSON(t *testing.T) {
	input := `not json`
	result := injectStreamUsage([]byte(input))
	if string(result) != input {
		t.Errorf("expected input returned unchanged, got %s", result)
	}
}

// --- Helper: resolveStreaming ---

func TestResolveStreaming(t *testing.T) {
	tr := true
	fa := false

	tests := []struct {
		endpoint string
		stream   *bool
		want     bool
	}{
		// Ollama native: streams by default
		{"/api/generate", nil, true},
		{"/api/generate", &tr, true},
		{"/api/generate", &fa, false},
		{"/api/chat", nil, true},
		// OpenAI: does not stream by default
		{"/v1/chat/completions", nil, false},
		{"/v1/chat/completions", &tr, true},
		{"/v1/chat/completions", &fa, false},
		// Anthropic: does not stream by default
		{"/v1/messages", nil, false},
		{"/v1/messages", &tr, true},
		{"/v1/messages", &fa, false},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%s/stream=%v", tt.endpoint, tt.stream)
		t.Run(name, func(t *testing.T) {
			got := resolveStreaming(tt.endpoint, tt.stream)
			if got != tt.want {
				t.Errorf("resolveStreaming(%q, %v) = %v, want %v", tt.endpoint, tt.stream, got, tt.want)
			}
		})
	}
}

// --- Accumulation test ---

func TestMetricsAccumulate(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(ollamaStats{
			Model:           "llama3",
			Done:            true,
			PromptEvalCount: 10,
			EvalCount:       20,
		})
	})

	p, reg := testProxy(t, backend)
	for range 3 {
		postJSON(t, p, "/api/generate", `{"model":"llama3","prompt":"hi","stream":false}`)
	}

	assertCounter(t, reg, "ollama_requests_total", 3, prometheus.Labels{"model": "llama3", "endpoint": "/api/generate"})
	assertCounter(t, reg, "ollama_prompt_tokens_total", 30, prometheus.Labels{"model": "llama3"})
	assertCounter(t, reg, "ollama_completion_tokens_total", 60, prometheus.Labels{"model": "llama3"})
}

func TestRequestDurationRecorded(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(ollamaStats{Model: "llama3", Done: true})
	})

	p, reg := testProxy(t, backend)
	postJSON(t, p, "/api/generate", `{"model":"llama3","prompt":"hi","stream":false}`)

	count := histogramCount(reg, "ollama_request_duration_seconds", prometheus.Labels{"model": "llama3", "endpoint": "/api/generate"})
	if count != 1 {
		t.Errorf("expected 1 histogram observation, got %d", count)
	}
}

func TestActiveRequestsReturnsToZero(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(ollamaStats{Model: "llama3", Done: true})
	})

	p, reg := testProxy(t, backend)
	postJSON(t, p, "/api/generate", `{"model":"llama3","prompt":"hi","stream":false}`)

	v := gaugeValue(reg, "ollama_active_requests", prometheus.Labels{"model": "llama3", "endpoint": "/api/generate"})
	if v != 0 {
		t.Errorf("expected active_requests=0 after completion, got %v", v)
	}
}

// --- Assertion helpers ---

func assertCounter(t *testing.T, reg *prometheus.Registry, name string, want float64, labels prometheus.Labels) {
	t.Helper()
	got := counterValue(reg, name, labels)
	if got != want {
		t.Errorf("%s%v = %v, want %v", name, labels, got, want)
	}
}

func assertGauge(t *testing.T, reg *prometheus.Registry, name string, want float64, labels prometheus.Labels) {
	t.Helper()
	got := gaugeValue(reg, name, labels)
	if got != want {
		t.Errorf("%s%v = %v, want %v", name, labels, got, want)
	}
}
