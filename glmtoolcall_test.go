package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

const glmToolsRequest = `{"model":"glm45-air-q4m:latest","stream":true,
"messages":[{"role":"user","content":"read it"}],
"tools":[
 {"type":"function","function":{"name":"read_file","description":"Read","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"startLine":{"type":"number"},"endLine":{"type":"number"}},"required":["filePath"]}}},
 {"type":"function","function":{"name":"list_dir","description":"List","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}},
 {"type":"function","function":{"name":"get_errors","description":"Errors","parameters":{}}}
]}`

func glmTools(t *testing.T) []requestToolDef {
	t.Helper()
	defs := parseRequestTools([]byte(glmToolsRequest))
	if len(defs) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(defs))
	}
	return defs
}

// chunk builds an /api/chat streaming chunk line.
func chunk(content, thinking string, done bool) string {
	msg := map[string]any{"role": "assistant", "content": content}
	if thinking != "" {
		msg["thinking"] = thinking
	}
	root := map[string]any{"model": "glm45-air-q4m:latest", "created_at": "2026-09-23T07:21:41Z", "message": msg, "done": done}
	if done {
		root["done_reason"] = "stop"
		root["eval_count"] = 55
		root["prompt_eval_count"] = 18911
		root["eval_duration"] = 2624460000
		root["prompt_eval_duration"] = 1286650000
		root["total_duration"] = 3911110000
	}
	b, _ := json.Marshal(root)
	return string(b)
}

// collect parses forwarded NDJSON lines into concatenated content, tool calls and the done flag.
func collect(t *testing.T, ndjson string) (content string, toolCalls []recoveredToolCall, sawDone bool, lines int) {
	t.Helper()
	for _, line := range strings.Split(strings.TrimSpace(ndjson), "\n") {
		if line == "" {
			continue
		}
		lines++
		var c struct {
			Message struct {
				Content   string              `json:"content"`
				ToolCalls []recoveredToolCall `json:"tool_calls"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("bad line %q: %v", line, err)
		}
		content += c.Message.Content
		toolCalls = append(toolCalls, c.Message.ToolCalls...)
		if c.Done {
			sawDone = true
		}
	}
	return
}

func runFilter(t *testing.T, lines []string) string {
	t.Helper()
	f := newGLMStreamFilter(glmTools(t), "/api/chat")
	var sb strings.Builder
	for _, l := range lines {
		for _, out := range f.process([]byte(l)) {
			sb.Write(out)
			sb.WriteByte('\n')
		}
	}
	return sb.String()
}

// The exact failure seen with GLM-4.5-Air: after thinking, the model omits
// <tool_call> and the call streams out as content, token by token.
func TestGLMRecoverBareToolCallStreaming(t *testing.T) {
	lines := []string{
		chunk("", "Let me", false),
		chunk("", " read it.", false),
		chunk("read", "", false),
		chunk("_file", "", false),
		chunk("<arg_key>", "", false),
		chunk("filePath", "", false),
		chunk("</arg_key>", "", false),
		chunk("<arg_value>", "", false),
		chunk("/home", "", false),
		chunk("/x/README.md", "", false),
		chunk("</arg_value>", "", false),
		chunk("<arg_key>", "", false),
		chunk("startLine", "", false),
		chunk("</arg_key>", "", false),
		chunk("<arg_value>", "", false),
		chunk("3", "", false),
		chunk("</arg_value>", "", false),
		chunk("</tool_call>", "", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, done, _ := collect(t, out)
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	if len(tcs) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %s", len(tcs), out)
	}
	tc := tcs[0]
	if tc.Function.Name != "read_file" {
		t.Fatalf("name = %q", tc.Function.Name)
	}
	if tc.Function.Arguments["filePath"] != "/home/x/README.md" {
		t.Fatalf("filePath = %v", tc.Function.Arguments["filePath"])
	}
	if tc.Function.Arguments["startLine"] != float64(3) {
		t.Fatalf("startLine = %#v (want number 3)", tc.Function.Arguments["startLine"])
	}
	if !strings.HasPrefix(tc.ID, "call_") {
		t.Fatalf("id = %q", tc.ID)
	}
	if !done {
		t.Fatal("final done chunk was not forwarded")
	}
	// Thinking chunks must pass through untouched.
	if !strings.Contains(out, `"thinking":"Let me"`) {
		t.Fatalf("thinking chunk missing from output: %s", out)
	}
}

// Content that merely starts with a tool name must be forwarded intact.
func TestGLMRecoverLeavesOrdinaryContentIntact(t *testing.T) {
	lines := []string{
		chunk("read", "", false),
		chunk("_file", "", false),
		chunk(" is the tool", "", false),
		chunk(" I would use.", "", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, done, _ := collect(t, out)
	if content != "read_file is the tool I would use." {
		t.Fatalf("content = %q", content)
	}
	if len(tcs) != 0 || !done {
		t.Fatalf("tcs=%d done=%v", len(tcs), done)
	}
}

// Content that diverges from every tool name on the first chunk is forwarded
// byte-for-byte.
func TestGLMRecoverPassthroughIsByteIdentical(t *testing.T) {
	lines := []string{
		chunk("", "thinking", false),
		chunk("Hello", "", false),
		chunk(" there", "", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	want := strings.Join(lines, "\n") + "\n"
	if out != want {
		t.Fatalf("output altered:\n got: %s\nwant: %s", out, want)
	}
}

// A real tool call emitted by Ollama is untouched.
func TestGLMRecoverRealToolCallPassthrough(t *testing.T) {
	real := `{"model":"m","created_at":"x","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"index":0,"name":"list_dir","arguments":{"path":"/"}}}]},"done":false}`
	lines := []string{chunk("", "t", false), real, chunk("", "", true)}
	out := runFilter(t, lines)
	if !strings.Contains(out, real) {
		t.Fatalf("real tool call line altered: %s", out)
	}
	_, tcs, _, _ := collect(t, out)
	if len(tcs) != 1 || tcs[0].ID != "call_1" {
		t.Fatalf("tcs = %+v", tcs)
	}
}

// Stream ends before </tool_call>: parse what we have, like Ollama's parser does.
func TestGLMRecoverUnterminatedAtDone(t *testing.T) {
	lines := []string{
		chunk("list_dir<arg_key>path</arg_key><arg_value>/tmp", "", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, done, _ := collect(t, out)
	if content != "" || len(tcs) != 1 || !done {
		t.Fatalf("content=%q tcs=%d done=%v out=%s", content, len(tcs), done, out)
	}
	if tcs[0].Function.Arguments["path"] != "/tmp" {
		t.Fatalf("args = %v", tcs[0].Function.Arguments)
	}
}

// Text after a recovered call (or whitespace before it) is kept as content.
func TestGLMRecoverWhitespaceAndTrailingText(t *testing.T) {
	lines := []string{
		chunk("\n", "", false),
		chunk("get_errors</tool_call>", "", false),
		chunk("\nDone.", "", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, _, _ := collect(t, out)
	if len(tcs) != 1 || tcs[0].Function.Name != "get_errors" || len(tcs[0].Function.Arguments) != 0 {
		t.Fatalf("tcs = %+v", tcs)
	}
	if content != "Done." {
		t.Fatalf("content = %q", content)
	}
}

func TestGLMRecoverNonStreaming(t *testing.T) {
	body := `{"model":"m","created_at":"x","message":{"role":"assistant","content":"read_file<arg_key>filePath</arg_key><arg_value>/a</arg_value></tool_call>","thinking":"hm"},"done":true,"done_reason":"stop","eval_count":5}`
	out := recoverGLMToolCallsInResponse([]byte(body), glmTools(t), "/api/chat", "", 0)
	var resp struct {
		Message struct {
			Content   string              `json:"content"`
			Thinking  string              `json:"thinking"`
			ToolCalls []recoveredToolCall `json:"tool_calls"`
		} `json:"message"`
		Done      bool `json:"done"`
		EvalCount int  `json:"eval_count"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "" || len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Function.Arguments["filePath"] != "/a" {
		t.Fatalf("bad repair: %s", out)
	}
	if resp.Message.Thinking != "hm" || !resp.Done || resp.EvalCount != 5 {
		t.Fatalf("unrelated fields altered: %s", out)
	}

	plain := `{"model":"m","message":{"role":"assistant","content":"Hello"},"done":true}`
	if got := recoverGLMToolCallsInResponse([]byte(plain), glmTools(t), "/api/chat", "", 0); string(got) != plain {
		t.Fatalf("plain response altered: %s", got)
	}
}

func TestGLMRecoverNoToolsIsNoop(t *testing.T) {
	if f := newGLMStreamFilter(nil, "/api/chat"); f != nil {
		t.Fatal("filter should be nil without tools")
	}
	body := `{"message":{"role":"assistant","content":"read_file<arg_key>x</arg_key></tool_call>"},"done":true}`
	if got := recoverGLMToolCallsInResponse([]byte(body), nil, "/api/chat", "", 0); string(got) != body {
		t.Fatal("altered without tools")
	}
}

func TestParseRequestTools(t *testing.T) {
	defs := glmTools(t)
	if defs[0].Name != "read_file" || defs[0].PropTypes["filePath"] != "string" || defs[0].PropTypes["startLine"] != "number" {
		t.Fatalf("defs[0] = %+v", defs[0])
	}
	if defs[2].Name != "get_errors" || len(defs[2].PropTypes) != 0 {
		t.Fatalf("defs[2] = %+v", defs[2])
	}
	if parseRequestTools([]byte(`{"model":"m"}`)) != nil {
		t.Fatal("expected nil without tools")
	}
}

// End-to-end through the proxy: streaming /api/chat with a broken tool call
// is repaired and metrics are still recorded from the final chunk.
func TestProxyRepairsBareGLMToolCallEndToEnd(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, l := range []string{
			chunk("", "thinking", false),
			chunk("list_dir", "", false),
			chunk("<arg_key>path</arg_key>", "", false),
			chunk("<arg_value>/srv</arg_value>", "", false),
			chunk("</tool_call>", "", false),
			chunk("", "", true),
		} {
			fmt.Fprintln(w, l)
		}
	})
	p, reg := testProxy(t, backend)
	rr := postJSON(t, p, "/api/chat", glmToolsRequest)
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d", rr.Code)
	}
	content, tcs, done, _ := collect(t, rr.Body.String())
	if content != "" || len(tcs) != 1 || tcs[0].Function.Name != "list_dir" || tcs[0].Function.Arguments["path"] != "/srv" || !done {
		t.Fatalf("content=%q tcs=%+v done=%v body=%s", content, tcs, done, rr.Body.String())
	}
	assertCounter(t, reg, "ollama_completion_tokens_total", 55, prometheus.Labels{"model": "glm45-air-q4m:latest"})
}

// Body dumps: written only while the dump directory exists; request body and
// every forwarded chunk are captured.
func TestBodyDumpWhenDirectoryExists(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, chunk("Hi", "", false))
		fmt.Fprintln(w, chunk("", "", true))
	})
	p, _ := testProxy(t, backend)
	dir := t.TempDir()
	p.dumpDir = dir

	rr := postJSON(t, p, "/api/chat", `{"model":"m","messages":[{"role":"user","content":"x"}]}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d", rr.Code)
	}
	reqs, _ := filepath.Glob(filepath.Join(dir, "*.req.json"))
	resps, _ := filepath.Glob(filepath.Join(dir, "*.resp.ndjson"))
	if len(reqs) != 1 || len(resps) != 1 {
		t.Fatalf("expected one req and one resp dump, got %v %v", reqs, resps)
	}
	reqBody, _ := os.ReadFile(reqs[0])
	if !strings.Contains(string(reqBody), `"content":"x"`) {
		t.Fatalf("request dump = %s", reqBody)
	}
	respBody, _ := os.ReadFile(resps[0])
	if string(respBody) != rr.Body.String() {
		t.Fatalf("response dump differs from what the client received:\n%s\n---\n%s", respBody, rr.Body.String())
	}

	// Removing the directory switches dumping off without a restart.
	os.RemoveAll(dir)
	postJSON(t, p, "/api/chat", `{"model":"m","messages":[]}`)
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("dump directory was recreated")
	}
}

// The exact case from the 2026-09-23 Copilot dump: the tool call is emitted
// inside the thinking block (no </think>), with a malformed
// "<arg_key>endLine<arg_value>50" (missing </arg_key>).
func TestGLMRecoverToolCallInsideThinking(t *testing.T) {
	lines := []string{
		chunk("", "I understand you want", false),
		chunk("", " help.\n\nLet me check what documentation exists:", false),
		chunk("", "<tool_call>", false),
		chunk("", "read", false),
		chunk("", "_file", false),
		chunk("", "<arg_key>", false),
		chunk("", "filePath", false),
		chunk("", "</arg_key>", false),
		chunk("", "<arg_value>", false),
		chunk("", "/home/leigh/README.md", false),
		chunk("", "</arg_value>", false),
		chunk("", "<arg_key>", false),
		chunk("", "startLine", false),
		chunk("", "</arg_key>", false),
		chunk("", "<arg_value>", false),
		chunk("", "1", false),
		chunk("", "</arg_value>", false),
		chunk("", "<arg_key>", false),
		chunk("", "endLine", false),
		chunk("", "<arg_value>", false),
		chunk("", "50", false),
		chunk("", "</arg_value>", false),
		chunk("", "</tool_call>", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, done, _ := collect(t, out)
	if content != "" {
		t.Fatalf("content = %q", content)
	}
	if len(tcs) != 1 || tcs[0].Function.Name != "read_file" {
		t.Fatalf("tcs = %+v\n%s", tcs, out)
	}
	args := tcs[0].Function.Arguments
	if args["filePath"] != "/home/leigh/README.md" || args["startLine"] != float64(1) || args["endLine"] != float64(50) {
		t.Fatalf("args = %#v", args)
	}
	if !done {
		t.Fatal("no done chunk")
	}
	// The thinking text before the call is preserved, the markup is not.
	th := collectThinking(t, out)
	if !strings.Contains(th, "Let me check what documentation exists:") || strings.Contains(th, "<arg_key>") || strings.Contains(th, "<tool_call>") {
		t.Fatalf("thinking = %q", th)
	}
}

// Same, but the tag itself arrives split across chunks and without <tool_call>
// (bare "read_file<arg_key>" inside thinking).
func TestGLMRecoverBareToolCallInsideThinkingSplitTags(t *testing.T) {
	lines := []string{
		chunk("", "Let me look:", false),
		chunk("", "read_file<arg", false),
		chunk("", "_key>filePath</arg_key><arg_value>/a</arg_value></tool", false),
		chunk("", "_call>", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, _, _ := collect(t, out)
	if content != "" || len(tcs) != 1 || tcs[0].Function.Arguments["filePath"] != "/a" {
		t.Fatalf("content=%q tcs=%+v\n%s", content, tcs, out)
	}
	// The tool name was already forwarded before <arg_key> arrived; that is
	// acceptable, but no tag markup may leak into thinking.
	if th := collectThinking(t, out); !strings.HasPrefix(th, "Let me look:") || strings.Contains(th, "<arg") || strings.Contains(th, "</tool") {
		t.Fatalf("thinking = %q", th)
	}
}

// Thinking with no tool call and no content: promoted to content so the
// client does not show an empty response.
func TestGLMThinkingOnlyPromotedToContent(t *testing.T) {
	lines := []string{
		chunk("", "The answer is", false),
		chunk("", " forty-two.", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	content, tcs, done, _ := collect(t, out)
	if content != "The answer is forty-two." || len(tcs) != 0 || !done {
		t.Fatalf("content=%q tcs=%d done=%v\n%s", content, len(tcs), done, out)
	}
}

// Ordinary thinking followed by ordinary content is forwarded byte-identical.
func TestGLMThinkingThenContentUntouched(t *testing.T) {
	lines := []string{
		chunk("", "Thinking about <things> and <tool", false),
		chunk("", "s> here.", false),
		chunk("Hello", "", false),
		chunk("", "", true),
	}
	out := runFilter(t, lines)
	if th := collectThinking(t, out); th != "Thinking about <things> and <tools> here." {
		t.Fatalf("thinking = %q", th)
	}
	content, tcs, done, _ := collect(t, out)
	if content != "Hello" || len(tcs) != 0 || !done {
		t.Fatalf("content=%q tcs=%d done=%v", content, len(tcs), done)
	}
	// The final line is byte-identical (nothing was ever withheld at that point).
	if !strings.Contains(out, lines[3]) {
		t.Fatalf("final chunk altered:\n%s", out)
	}
}

func TestToolCallParserToleratesMissingTags(t *testing.T) {
	p := &toolCallParser{tools: glmTools(t), endpoint: "/api/chat"}
	cases := map[string]map[string]any{
		"read_file<arg_key>filePath</arg_key><arg_value>/a</arg_value>":                                         {"filePath": "/a"},
		"read_file<arg_key>filePath<arg_value>/a</arg_value>":                                                   {"filePath": "/a"},                          // missing </arg_key>
		"read_file<arg_key>filePath</arg_key>/a</arg_value>":                                                    {"filePath": "/a"},                          // missing <arg_value>
		"read_file<arg_key>filePath</arg_key><arg_value>/a<arg_key>startLine</arg_key><arg_value>2</arg_value>": {"filePath": "/a", "startLine": float64(2)}, // missing </arg_value>
		"get_errors": {},
	}
	for raw, want := range cases {
		tc, ok := p.parse(raw, "test")
		if !ok {
			t.Fatalf("%q: not parsed", raw)
		}
		if len(tc.Function.Arguments) != len(want) {
			t.Fatalf("%q: args = %#v, want %#v", raw, tc.Function.Arguments, want)
		}
		for k, v := range want {
			if tc.Function.Arguments[k] != v {
				t.Fatalf("%q: arg %s = %#v, want %#v", raw, k, tc.Function.Arguments[k], v)
			}
		}
	}
	if _, ok := p.parse("unknown_tool<arg_key>x</arg_key>", "test"); ok {
		t.Fatal("unknown tool must not parse")
	}
}

func collectThinking(t *testing.T, ndjson string) string {
	t.Helper()
	var sb strings.Builder
	for _, line := range strings.Split(strings.TrimSpace(ndjson), "\n") {
		if line == "" {
			continue
		}
		var c struct {
			Message struct {
				Thinking string `json:"thinking"`
			} `json:"message"`
		}
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("bad line %q: %v", line, err)
		}
		sb.WriteString(c.Message.Thinking)
	}
	return sb.String()
}

// Loop breaker: the request history already holds three identical read_file
// calls (differing only in line numbers); a fourth is replaced by a notice.
const loopingRequest = `{"model":"glm45-air-q4m:latest","stream":true,
"messages":[
 {"role":"user","content":"update from git"},
 {"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"name":"read_file","arguments":{"filePath":"/r/.git/logs/HEAD","startLine":-1,"endLine":1}}}]},
 {"role":"tool","content":"abc fetch","tool_call_id":"c1"},
 {"role":"assistant","content":"","tool_calls":[{"id":"c2","function":{"name":"read_file","arguments":{"filePath":"/r/.git/logs/HEAD","startLine":-2,"endLine":1}}}]},
 {"role":"tool","content":"abc fetch","tool_call_id":"c2"},
 {"role":"assistant","content":"","tool_calls":[{"id":"c3","function":{"name":"read_file","arguments":{"filePath":"/r/.git/logs/HEAD","startLine":-3,"endLine":1}}}]},
 {"role":"tool","content":"abc fetch","tool_call_id":"c3"}
],
"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"startLine":{"type":"number"},"endLine":{"type":"number"}}}}}]}`

func TestDetectRepeatedToolCall(t *testing.T) {
	key, n := detectRepeatedToolCall([]byte(loopingRequest))
	if n != 3 || key != "read_file|filePath=head" {
		t.Fatalf("key=%q n=%d", key, n)
	}
	if _, n := detectRepeatedToolCall([]byte(glmToolsRequest)); n != 0 {
		t.Fatalf("no loop expected, got %d", n)
	}
}

func TestLoopBreakerReplacesNativeToolCall(t *testing.T) {
	native := `{"model":"m","created_at":"x","message":{"role":"assistant","content":"","tool_calls":[{"id":"c4","function":{"index":0,"name":"read_file","arguments":{"filePath":"/r/.git/logs/HEAD","startLine":-4,"endLine":1}}}]},"done":false}`
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, native)
		fmt.Fprintln(w, chunk("", "", true))
	})
	p, _ := testProxy(t, backend)
	rr := postJSON(t, p, "/api/chat", loopingRequest)
	content, tcs, done, _ := collect(t, rr.Body.String())
	if len(tcs) != 0 || !done || !strings.Contains(content, "Stopped a tool-call loop") || !strings.Contains(content, "4 times") {
		t.Fatalf("content=%q tcs=%d done=%v\n%s", content, len(tcs), done, rr.Body.String())
	}
}

func TestLoopBreakerReplacesRecoveredToolCall(t *testing.T) {
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, chunk("", "Let me check:<tool_call>read_file<arg_key>filePath</arg_key><arg_value>/r/.git/logs/HEAD</arg_value><arg_key>startLine</arg_key><arg_value>-4</arg_value></tool_call>", false))
		fmt.Fprintln(w, chunk("", "", true))
	})
	p, _ := testProxy(t, backend)
	rr := postJSON(t, p, "/api/chat", loopingRequest)
	content, tcs, _, _ := collect(t, rr.Body.String())
	if len(tcs) != 0 || !strings.Contains(content, "Stopped a tool-call loop") {
		t.Fatalf("content=%q tcs=%d\n%s", content, len(tcs), rr.Body.String())
	}
}

// A different call is not a loop and passes through.
func TestLoopBreakerLetsNewCallThrough(t *testing.T) {
	native := `{"model":"m","created_at":"x","message":{"role":"assistant","content":"","tool_calls":[{"id":"c4","function":{"index":0,"name":"read_file","arguments":{"filePath":"/r/README.md","startLine":1,"endLine":50}}}]},"done":false}`
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, native)
		fmt.Fprintln(w, chunk("", "", true))
	})
	p, _ := testProxy(t, backend)
	rr := postJSON(t, p, "/api/chat", loopingRequest)
	content, tcs, _, _ := collect(t, rr.Body.String())
	if len(tcs) != 1 || content != "" {
		t.Fatalf("content=%q tcs=%d", content, len(tcs))
	}
}

// Completely empty response (model emitted EOS immediately): a notice is shown.
func TestEmptyResponseNotice(t *testing.T) {
	out := runFilter(t, []string{chunk("", "", true)})
	content, tcs, done, _ := collect(t, out)
	if len(tcs) != 0 || !done || !strings.Contains(content, "empty response") {
		t.Fatalf("content=%q done=%v\n%s", content, done, out)
	}
}

// A garbled directory in the path (as the model actually produced) still
// counts as the same looping call.
func TestLoopBreakerToleratesGarbledDirectory(t *testing.T) {
	native := `{"model":"m","created_at":"x","message":{"role":"assistant","content":"","tool_calls":[{"id":"c4","function":{"index":0,"name":"read_file","arguments":{"filePath":"/r/.git/lgs/HEAD","startLine":-4,"endLine":1}}}]},"done":false}`
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, native)
		fmt.Fprintln(w, chunk("", "", true))
	})
	p, _ := testProxy(t, backend)
	rr := postJSON(t, p, "/api/chat", loopingRequest)
	content, tcs, _, _ := collect(t, rr.Body.String())
	if len(tcs) != 0 || !strings.Contains(content, "Stopped a tool-call loop") {
		t.Fatalf("content=%q tcs=%d", content, len(tcs))
	}
}
