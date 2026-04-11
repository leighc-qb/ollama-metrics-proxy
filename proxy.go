package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Ollama native API response fields present in the final chunk (done=true).
type ollamaStats struct {
	Model              string `json:"model"`
	Done               bool   `json:"done"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
}

// OpenAI-compatible /v1/chat/completions response.
type openAIResponse struct {
	Model string       `json:"model"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Anthropic /v1/messages response.
type anthropicResponse struct {
	Model string          `json:"model"`
	Usage *anthropicUsage `json:"usage,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// anthropicStreamEvent covers the SSE event types we care about.
type anthropicStreamEvent struct {
	Type    string             `json:"type"`
	Message *anthropicResponse `json:"message,omitempty"`
	Usage   *anthropicUsage    `json:"usage,omitempty"`
}

// requestInfo extracts fields we need from any incoming request body.
type requestInfo struct {
	Model  string `json:"model"`
	Stream *bool  `json:"stream"`
}

// proxy is a reverse proxy that transparently forwards requests to an Ollama
// backend while capturing inference metrics from the responses.
type proxy struct {
	ollamaURL *url.URL
	reverse   *httputil.ReverseProxy
	client    *http.Client

	requestsTotal         *prometheus.CounterVec
	promptTokensTotal     *prometheus.CounterVec
	completionTokensTotal *prometheus.CounterVec
	requestDuration       *prometheus.HistogramVec
	promptEvalDuration    *prometheus.CounterVec
	evalDuration          *prometheus.CounterVec
	modelLoadDuration     *prometheus.CounterVec
	tokensPerSecond       *prometheus.GaugeVec
	activeRequests        *prometheus.GaugeVec
}

const maxScanBuf = 1024 * 1024 // 1 MiB line buffer for streaming responses

func newProxy(ollamaURL *url.URL, reg prometheus.Registerer) *proxy {
	rp := httputil.NewSingleHostReverseProxy(ollamaURL)
	rp.Transport = &http.Transport{
		ResponseHeaderTimeout: 0,
		IdleConnTimeout:       0,
	}

	p := &proxy{
		ollamaURL: ollamaURL,
		reverse:   rp,
		client:    &http.Client{Timeout: 0},

		requestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_requests_total",
			Help: "Total completed inference requests.",
		}, []string{"model", "endpoint"}),
		promptTokensTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_prompt_tokens_total",
			Help: "Total prompt tokens processed.",
		}, []string{"model"}),
		completionTokensTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_completion_tokens_total",
			Help: "Total completion tokens generated.",
		}, []string{"model"}),
		requestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "ollama_request_duration_seconds",
			Help:    "End-to-end request duration in seconds.",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600},
		}, []string{"model", "endpoint"}),
		promptEvalDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_prompt_eval_seconds_total",
			Help: "Total time evaluating prompts in seconds (Ollama native only).",
		}, []string{"model"}),
		evalDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_token_generation_seconds_total",
			Help: "Total time generating tokens in seconds (Ollama native only).",
		}, []string{"model"}),
		modelLoadDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_model_load_seconds_total",
			Help: "Total time loading models in seconds (Ollama native only).",
		}, []string{"model"}),
		tokensPerSecond: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_tokens_per_second",
			Help: "Most recent token generation speed (Ollama native only).",
		}, []string{"model"}),
		activeRequests: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_active_requests",
			Help: "Currently in-flight inference requests.",
		}, []string{"model", "endpoint"}),
	}

	reg.MustRegister(
		p.requestsTotal, p.promptTokensTotal, p.completionTokensTotal,
		p.requestDuration, p.promptEvalDuration, p.evalDuration,
		p.modelLoadDuration, p.tokensPerSecond, p.activeRequests,
	)

	return p
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/api/generate", "/api/chat", "/v1/chat/completions", "/v1/messages":
		p.handleInference(w, r)
	default:
		p.reverse.ServeHTTP(w, r)
	}
}

func (p *proxy) handleInference(w http.ResponseWriter, r *http.Request) {
	endpoint := r.URL.Path

	body, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadGateway)
		return
	}

	var info requestInfo
	json.Unmarshal(body, &info)

	model := info.Model
	if model == "" {
		model = "unknown"
	}

	streaming := resolveStreaming(endpoint, info.Stream)

	// For OpenAI streaming, inject stream_options so Ollama returns usage.
	if endpoint == "/v1/chat/completions" && streaming {
		body = injectStreamUsage(body)
	}

	p.activeRequests.WithLabelValues(model, endpoint).Inc()
	start := time.Now()
	defer func() {
		p.activeRequests.WithLabelValues(model, endpoint).Dec()
		p.requestDuration.WithLabelValues(model, endpoint).Observe(time.Since(start).Seconds())
	}()

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, p.ollamaURL.String()+endpoint, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "failed to create upstream request", http.StatusBadGateway)
		return
	}
	proxyReq.Header = r.Header.Clone()
	proxyReq.Header.Del("Content-Length") // let Go recalculate from body

	resp, err := p.client.Do(proxyReq)
	if err != nil {
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		io.Copy(w, resp.Body)
		return
	}

	flusher, canFlush := w.(http.Flusher)
	ctx := streamContext{w: w, flusher: flusher, canFlush: canFlush}

	switch endpoint {
	case "/v1/chat/completions":
		p.handleOpenAI(ctx, resp.Body, streaming, model, endpoint)
	case "/v1/messages":
		p.handleAnthropic(ctx, resp.Body, streaming, model, endpoint)
	default:
		p.handleOllama(ctx, resp.Body, streaming, model, endpoint)
	}
}

// streamContext bundles the response writer and optional flusher.
type streamContext struct {
	w        http.ResponseWriter
	flusher  http.Flusher
	canFlush bool
}

func (sc streamContext) writeLine(line string) {
	sc.w.Write([]byte(line))
	sc.w.Write([]byte("\n"))
	if sc.canFlush {
		sc.flusher.Flush()
	}
}

// resolveStreaming determines whether a request is streaming based on the
// endpoint type and the explicit stream field in the request body.
// Ollama native endpoints stream by default; OpenAI and Anthropic do not.
func resolveStreaming(endpoint string, stream *bool) bool {
	switch endpoint {
	case "/v1/chat/completions", "/v1/messages":
		return stream != nil && *stream
	default:
		return stream == nil || *stream
	}
}

func copyHeaders(w http.ResponseWriter, resp *http.Response) {
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
}

func newScanner(r io.Reader) *bufio.Scanner {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, maxScanBuf), maxScanBuf)
	return s
}

// --- Ollama native handler (/api/generate, /api/chat) ---

func (p *proxy) handleOllama(ctx streamContext, body io.Reader, streaming bool, model, endpoint string) {
	if !streaming {
		respBody, err := io.ReadAll(body)
		if err != nil {
			return
		}
		ctx.w.Write(respBody)
		var stats ollamaStats
		if json.Unmarshal(respBody, &stats) == nil {
			p.recordOllamaMetrics(model, endpoint, &stats)
		}
		return
	}

	scanner := newScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		ctx.writeLine(line)

		var stats ollamaStats
		if json.Unmarshal([]byte(line), &stats) == nil && stats.Done {
			p.recordOllamaMetrics(model, endpoint, &stats)
		}
	}
}

func (p *proxy) recordOllamaMetrics(reqModel, endpoint string, stats *ollamaStats) {
	model := stats.Model
	if model == "" {
		model = reqModel
	}

	p.requestsTotal.WithLabelValues(model, endpoint).Inc()
	p.promptTokensTotal.WithLabelValues(model).Add(float64(stats.PromptEvalCount))
	p.completionTokensTotal.WithLabelValues(model).Add(float64(stats.EvalCount))

	if stats.PromptEvalDuration > 0 {
		p.promptEvalDuration.WithLabelValues(model).Add(float64(stats.PromptEvalDuration) / 1e9)
	}
	if stats.EvalDuration > 0 {
		p.evalDuration.WithLabelValues(model).Add(float64(stats.EvalDuration) / 1e9)
		tps := float64(stats.EvalCount) / (float64(stats.EvalDuration) / 1e9)
		p.tokensPerSecond.WithLabelValues(model).Set(tps)
	}
	if stats.LoadDuration > 0 {
		p.modelLoadDuration.WithLabelValues(model).Add(float64(stats.LoadDuration) / 1e9)
	}
}

// --- OpenAI-compatible handler (/v1/chat/completions) ---

func (p *proxy) handleOpenAI(ctx streamContext, body io.Reader, streaming bool, model, endpoint string) {
	if !streaming {
		respBody, err := io.ReadAll(body)
		if err != nil {
			return
		}
		ctx.w.Write(respBody)
		var resp openAIResponse
		if json.Unmarshal(respBody, &resp) == nil && resp.Usage != nil {
			p.recordOpenAIMetrics(model, endpoint, &resp)
		}
		return
	}

	scanner := newScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		ctx.writeLine(line)

		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var resp openAIResponse
		if json.Unmarshal([]byte(data), &resp) == nil && resp.Usage != nil {
			p.recordOpenAIMetrics(model, endpoint, &resp)
		}
	}
}

func (p *proxy) recordOpenAIMetrics(reqModel, endpoint string, resp *openAIResponse) {
	model := resp.Model
	if model == "" {
		model = reqModel
	}
	p.requestsTotal.WithLabelValues(model, endpoint).Inc()
	p.promptTokensTotal.WithLabelValues(model).Add(float64(resp.Usage.PromptTokens))
	p.completionTokensTotal.WithLabelValues(model).Add(float64(resp.Usage.CompletionTokens))
}

// --- Anthropic handler (/v1/messages) ---

func (p *proxy) handleAnthropic(ctx streamContext, body io.Reader, streaming bool, model, endpoint string) {
	if !streaming {
		respBody, err := io.ReadAll(body)
		if err != nil {
			return
		}
		ctx.w.Write(respBody)
		var resp anthropicResponse
		if json.Unmarshal(respBody, &resp) == nil && resp.Usage != nil {
			p.recordAnthropicMetrics(model, endpoint, resp.Model, resp.Usage.InputTokens, resp.Usage.OutputTokens)
		}
		return
	}

	var inputTokens, outputTokens int
	var respModel string

	scanner := newScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		ctx.writeLine(line)

		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event anthropicStreamEvent
		if json.Unmarshal([]byte(data), &event) != nil {
			continue
		}
		switch event.Type {
		case "message_start":
			if event.Message != nil {
				respModel = event.Message.Model
				if event.Message.Usage != nil {
					inputTokens = event.Message.Usage.InputTokens
				}
			}
		case "message_delta":
			if event.Usage != nil {
				outputTokens = event.Usage.OutputTokens
			}
		case "message_stop":
			p.recordAnthropicMetrics(model, endpoint, respModel, inputTokens, outputTokens)
		}
	}
}

func (p *proxy) recordAnthropicMetrics(reqModel, endpoint, respModel string, inputTokens, outputTokens int) {
	model := respModel
	if model == "" {
		model = reqModel
	}
	p.requestsTotal.WithLabelValues(model, endpoint).Inc()
	p.promptTokensTotal.WithLabelValues(model).Add(float64(inputTokens))
	p.completionTokensTotal.WithLabelValues(model).Add(float64(outputTokens))
}

// --- Helpers ---

// injectStreamUsage adds stream_options.include_usage to an OpenAI request
// body so the server returns token counts in the final SSE chunk.
func injectStreamUsage(body []byte) []byte {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	raw["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	out, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return out
}
