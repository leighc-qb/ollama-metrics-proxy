package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"log"
	"mime"
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

	// dumpDir, when set and existing, receives /api/chat request/response dumps (see dump.go).
	dumpDir string
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

	var dump *bodyDump
	if endpoint == "/api/chat" {
		dump = p.newBodyDump(endpoint, body)
		if dump != nil {
			defer dump.Close()
		}
	}

	model := info.Model
	if model == "" {
		model = "unknown"
	}

	if r.Method == http.MethodPost && isJSONContentType(r) &&
		(endpoint == "/api/chat" || endpoint == "/v1/chat/completions") {
		var repaired int
		body, repaired = repairEmptyAssistantMessages(body)
		if repaired > 0 {
			log.Printf("copilot_null_content_repaired path=%s count=%d", endpoint, repaired)
		}
	}

	// Tool definitions are needed to recover GLM tool calls whose <tool_call>
	// tag the model dropped (see glmtoolcall.go). Ollama accepts JSON bodies
	// regardless of Content-Type, so do not gate this on the header.
	var requestTools []requestToolDef
	var loopKey string
	var loopRepeats int
	if r.Method == http.MethodPost && endpoint == "/api/chat" {
		requestTools = parseRequestTools(body)
		loopKey, loopRepeats = detectRepeatedToolCall(body)
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

	upstreamURL := *p.ollamaURL
	upstreamURL.Path = strings.TrimRight(upstreamURL.Path, "/") + endpoint
	upstreamURL.RawPath = ""
	upstreamURL.RawQuery = r.URL.RawQuery
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL.String(), bytes.NewReader(body))
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
		if dump != nil {
			io.Copy(io.MultiWriter(w, dump), resp.Body)
		} else {
			io.Copy(w, resp.Body)
		}
		return
	}

	flusher, canFlush := w.(http.Flusher)
	ctx := streamContext{w: w, flusher: flusher, canFlush: canFlush}
	if dump != nil {
		ctx.tee = dump
	}

	switch endpoint {
	case "/v1/chat/completions":
		p.handleOpenAI(ctx, resp.Body, streaming, model, endpoint)
	case "/v1/messages":
		p.handleAnthropic(ctx, resp.Body, streaming, model, endpoint)
	default:
		p.handleOllama(ctx, resp.Body, streaming, model, endpoint, requestTools, loopKey, loopRepeats)
	}
}

// streamContext bundles the response writer and optional flusher.
type streamContext struct {
	w        http.ResponseWriter
	flusher  http.Flusher
	canFlush bool
	tee      io.Writer // optional copy of everything written to the client
}

func (sc streamContext) write(b []byte) {
	sc.w.Write(b)
	if sc.tee != nil {
		sc.tee.Write(b)
	}
}

func (sc streamContext) writeLine(line string) {
	sc.write([]byte(line))
	sc.write([]byte("\n"))
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

func (p *proxy) handleOllama(ctx streamContext, body io.Reader, streaming bool, model, endpoint string, tools []requestToolDef, loopKey string, loopRepeats int) {
	if !streaming {
		respBody, err := io.ReadAll(body)
		if err != nil {
			return
		}
		if endpoint == "/api/chat" {
			respBody = recoverGLMToolCallsInResponse(respBody, tools, endpoint, loopKey, loopRepeats)
		}
		ctx.write(respBody)
		var stats ollamaStats
		if json.Unmarshal(respBody, &stats) == nil {
			p.recordOllamaMetrics(model, endpoint, &stats)
		}
		return
	}

	var filter *glmStreamFilter
	if endpoint == "/api/chat" {
		filter = newGLMStreamFilter(tools, endpoint) // nil when the request has no tools
		if filter != nil {
			filter.setLoopGuard(loopKey, loopRepeats)
		}
	}

	scanner := newScanner(body)
	for scanner.Scan() {
		line := scanner.Bytes()

		var stats ollamaStats
		if json.Unmarshal(line, &stats) == nil && stats.Done {
			p.recordOllamaMetrics(model, endpoint, &stats)
		}

		if filter == nil {
			ctx.writeLine(string(line))
			continue
		}
		for _, out := range filter.process(line) {
			ctx.writeLine(string(out))
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

func isJSONContentType(r *http.Request) bool {
	contentType := r.Header.Get("Content-Type")
	if contentType == "" {
		return true
	}

	mediaType, _, err := mime.ParseMediaType(contentType)
	return err == nil && (mediaType == "application/json" || strings.HasSuffix(mediaType, "+json"))
}

// repairEmptyAssistantMessages replaces only the invalid empty assistant turns
// that Ollama rejects, leaving assistant tool-call messages untouched.
func repairEmptyAssistantMessages(body []byte) ([]byte, int) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return body, 0
	}

	rawMessages, ok := root["messages"]
	if !ok {
		return body, 0
	}

	var messages []json.RawMessage
	if err := json.Unmarshal(rawMessages, &messages); err != nil || messages == nil {
		return body, 0
	}

	changed := 0
	for i, rawMessage := range messages {
		var message map[string]json.RawMessage
		if err := json.Unmarshal(rawMessage, &message); err != nil || message == nil {
			continue
		}

		var role string
		rawRole, ok := message["role"]
		if !ok || json.Unmarshal(rawRole, &role) != nil || role != "assistant" {
			continue
		}

		rawContent, ok := message["content"]
		if !ok || !isJSONNull(rawContent) {
			continue
		}

		if rawToolCalls, ok := message["tool_calls"]; ok && !isNullOrEmptyJSONArray(rawToolCalls) {
			continue
		}
		if rawFunctionCall, ok := message["function_call"]; ok && !isJSONNull(rawFunctionCall) {
			continue
		}

		message["content"] = json.RawMessage(`""`)
		repairedMessage, err := json.Marshal(message)
		if err != nil {
			return body, 0
		}
		messages[i] = repairedMessage
		changed++
	}

	if changed == 0 {
		return body, 0
	}

	repairedMessages, err := json.Marshal(messages)
	if err != nil {
		return body, 0
	}
	root["messages"] = repairedMessages
	repairedBody, err := json.Marshal(root)
	if err != nil {
		return body, 0
	}
	return repairedBody, changed
}

func isJSONNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

func isNullOrEmptyJSONArray(raw json.RawMessage) bool {
	if isJSONNull(raw) {
		return true
	}

	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		return false
	}
	return values != nil && len(values) == 0
}

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
