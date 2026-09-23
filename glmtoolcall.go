package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log"
	"regexp"
	"strings"
	"unicode"
)

// GLM-4.x models (GLM-4.5-Air in particular) produce tool calls that Ollama's
// glm parser cannot recognise, in two ways observed with Copilot Chat traffic:
//
//  1. The opening <tool_call> tag is omitted right after </think>:
//     read_file<arg_key>filePath</arg_key><arg_value>/x</arg_value></tool_call>
//     streams out as plain assistant content.
//  2. The whole call is emitted inside the thinking block, before any </think>:
//     "...Let me check:<tool_call>read_file<arg_key>...</tool_call>" and the
//     stream ends. Everything is classified as thinking; content stays empty.
//
// Copilot only forwards content and tool_calls, so both cases surface as
// "no response was returned" or as raw markup. This file repairs both on the
// /api/chat response path, tolerating the missing/mismatched <arg_key> and
// <arg_value> tags the model also produces. As a last resort, a response that
// ends with thinking only (no content, no tool call) has its thinking promoted
// to content so the client at least shows the model's answer.

const (
	glmArgKeyOpen   = "<arg_key>"
	glmArgKeyClose  = "</arg_key>"
	glmArgValOpen   = "<arg_value>"
	glmArgValClose  = "</arg_value>"
	glmToolClose    = "</tool_call>"
	glmToolOpenTag  = "<tool_call>"
	maxThinkingKeep = 256 * 1024 // cap on thinking retained for the promote-to-content fallback
)

var glmArgTagRe = regexp.MustCompile(`<arg_key>|</arg_key>|<arg_value>|</arg_value>`)

// requestToolDef is what the recoverer needs to know about one request tool.
type requestToolDef struct {
	Name      string
	PropTypes map[string]string // property name -> JSON schema type ("string", "integer", ...)
}

// parseRequestTools extracts the tool definitions from a /api/chat request body.
// It returns nil when the request has no usable tools.
func parseRequestTools(body []byte) []requestToolDef {
	var req struct {
		Tools []struct {
			Function struct {
				Name       string `json:"name"`
				Parameters struct {
					Properties map[string]struct {
						Type json.RawMessage `json:"type"`
					} `json:"properties"`
				} `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil
	}
	var defs []requestToolDef
	for _, t := range req.Tools {
		if t.Function.Name == "" {
			continue
		}
		def := requestToolDef{Name: t.Function.Name, PropTypes: map[string]string{}}
		for name, prop := range t.Function.Parameters.Properties {
			var typ string
			if json.Unmarshal(prop.Type, &typ) == nil {
				def.PropTypes[name] = typ
			}
		}
		defs = append(defs, def)
	}
	return defs
}

// recoveredToolCall mirrors Ollama's tool_calls entry shape.
type recoveredToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Index     int            `json:"index"`
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"function"`
}

// toolCallParser turns "name<arg_key>k</arg_key><arg_value>v</arg_value>..."
// (without the surrounding <tool_call> tags) into structured calls, keeping a
// running index like Ollama does.
type toolCallParser struct {
	tools    []requestToolDef
	index    int
	endpoint string
}

func (p *toolCallParser) lookup(name string) *requestToolDef {
	for i := range p.tools {
		if p.tools[i].Name == name {
			return &p.tools[i]
		}
	}
	return nil
}

// parse is tolerant of the tag mistakes GLM makes: a missing </arg_key>, a
// missing <arg_value>, or a missing </arg_value>. It follows the expected
// key/value cycle, taking text up to the next recognised tag.
func (p *toolCallParser) parse(raw string, where string) (recoveredToolCall, bool) {
	raw = strings.TrimSpace(raw)
	nameEnd := strings.Index(raw, glmArgKeyOpen)
	if nameEnd < 0 {
		nameEnd = len(raw)
	}
	name := strings.TrimSpace(raw[:nameEnd])
	def := p.lookup(name)
	if def == nil {
		return recoveredToolCall{}, false
	}

	args := map[string]any{}
	rest := raw[nameEnd:]
	// Tokenise into alternating text / tag pieces.
	type piece struct {
		tag  string
		text string
	}
	var pieces []piece
	for {
		loc := glmArgTagRe.FindStringIndex(rest)
		if loc == nil {
			if rest != "" {
				pieces = append(pieces, piece{text: rest})
			}
			break
		}
		if loc[0] > 0 {
			pieces = append(pieces, piece{text: rest[:loc[0]]})
		}
		pieces = append(pieces, piece{tag: rest[loc[0]:loc[1]]})
		rest = rest[loc[1]:]
	}
	// Walk the cycle: key text follows <arg_key>, value text follows <arg_value>
	// (or follows the key when <arg_value> is missing).
	const (
		wantKeyOpen = iota
		inKey
		wantValOpen
		inVal
	)
	state := wantKeyOpen
	var key, val strings.Builder
	flush := func() {
		k := strings.TrimSpace(key.String())
		if k != "" {
			args[k] = coerceGLMArg(strings.TrimSpace(val.String()), def.PropTypes[k])
		}
		key.Reset()
		val.Reset()
	}
	for _, pc := range pieces {
		switch {
		case pc.tag == glmArgKeyOpen:
			if state == inVal || state == wantValOpen {
				flush()
			}
			state = inKey
		case pc.tag == glmArgKeyClose:
			if state == inKey {
				state = wantValOpen
			}
		case pc.tag == glmArgValOpen:
			state = inVal
		case pc.tag == glmArgValClose:
			if state == inVal || state == wantValOpen {
				flush()
			}
			state = wantKeyOpen
		default: // text
			switch state {
			case inKey:
				key.WriteString(pc.text)
			case wantValOpen:
				// Missing <arg_value>: this text is the value.
				val.WriteString(pc.text)
				state = inVal
			case inVal:
				val.WriteString(pc.text)
			}
		}
	}
	if state == inVal || state == wantValOpen {
		flush()
	}

	var tc recoveredToolCall
	tc.ID = "call_" + randomHex(8)
	tc.Function.Index = p.index
	tc.Function.Name = name
	tc.Function.Arguments = args
	p.index++
	log.Printf("glm_toolcall_recovered path=%s tool=%s from=%s", p.endpoint, name, where)
	return tc, true
}

// coerceGLMArg converts a raw argument string according to the declared schema
// type. Strings stay strings; everything else is parsed as JSON when possible.
func coerceGLMArg(val, typ string) any {
	if typ == "string" {
		return val
	}
	var v any
	if err := json.Unmarshal([]byte(val), &v); err == nil {
		return v
	}
	return val
}

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return strings.Repeat("0", 2*n)
	}
	return hex.EncodeToString(b)
}

// --- Case 1: bare tool call at the start of content -------------------------

// A bare tool call starts with the tool name followed either by the first
// <arg_key> or, for a parameterless tool, directly by </tool_call>.
var glmToolStartSuffixes = []string{glmArgKeyOpen, glmToolClose}

// contentScanner withholds content until it is clear whether it is a bare tool
// call or ordinary text.
type contentScanner struct {
	parser    *toolCallParser
	pending   strings.Builder
	capturing bool
	trimNext  bool // drop whitespace directly after a recovered call, as Ollama's parser does
}

// feedResult is the output of one feed: text safe to forward and any recovered
// tool calls, in order. withheld reports whether some text is still held back.
type feedResult struct {
	text      string
	toolCalls []recoveredToolCall
	withheld  bool
}

func (r *contentScanner) feed(content string, done bool) feedResult {
	var out feedResult
	r.pending.WriteString(content)

	for {
		acc := r.pending.String()
		if acc == "" {
			break
		}

		if r.capturing {
			idx := strings.Index(acc, glmToolClose)
			if idx < 0 {
				if !done {
					out.withheld = true
					return out
				}
				// Stream ended inside a bare tool call: parse what we have.
				if tc, ok := r.parser.parse(acc, "content"); ok {
					out.toolCalls = append(out.toolCalls, tc)
				} else {
					out.text += acc
				}
				r.pending.Reset()
				r.capturing = false
				break
			}
			raw, rest := acc[:idx], acc[idx+len(glmToolClose):]
			r.pending.Reset()
			r.pending.WriteString(rest)
			r.capturing = false
			if tc, ok := r.parser.parse(raw, "content"); ok {
				out.toolCalls = append(out.toolCalls, tc)
				r.trimNext = true
			} else {
				out.text += raw + glmToolClose
			}
			continue
		}

		trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
		if trimmed == "" {
			// Only whitespace so far: hold it, it may precede a bare tool call.
			if done {
				if !r.trimNext {
					out.text += acc
				}
				r.pending.Reset()
			} else {
				out.withheld = true
			}
			break
		}
		if r.trimNext {
			acc = trimmed
			r.trimNext = false
		}
		if r.matchesToolStart(trimmed) {
			r.pending.Reset()
			r.pending.WriteString(trimmed)
			r.capturing = true
			continue
		}
		if !done && r.couldBecomeToolStart(trimmed) {
			out.withheld = true
			return out
		}
		// Ordinary content.
		out.text += acc
		r.pending.Reset()
		break
	}
	return out
}

func (r *contentScanner) matchesToolStart(s string) bool {
	for _, t := range r.parser.tools {
		for _, suffix := range glmToolStartSuffixes {
			if strings.HasPrefix(s, t.Name+suffix) {
				return true
			}
		}
	}
	return false
}

func (r *contentScanner) couldBecomeToolStart(s string) bool {
	for _, t := range r.parser.tools {
		for _, suffix := range glmToolStartSuffixes {
			if strings.HasPrefix(t.Name+suffix, s) {
				return true
			}
		}
	}
	return false
}

// --- Case 2: tool call embedded in thinking ---------------------------------

// thinkingScanner passes thinking through but extracts any <tool_call>...
// </tool_call> (or bare name<arg_key>...) it finds inside it.
type thinkingScanner struct {
	parser     *toolCallParser
	pending    strings.Builder
	capturing  bool
	namePrefix string // tool name already forwarded as thinking before <arg_key> was seen
	recentTail string // last forwarded thinking, to recognise a bare tool name
}

const thinkingTailKeep = 128

func (r *thinkingScanner) remember(s string) {
	r.recentTail += s
	if len(r.recentTail) > thinkingTailKeep {
		r.recentTail = r.recentTail[len(r.recentTail)-thinkingTailKeep:]
	}
}

// toolStartInThinking finds where a tool call begins in acc: at an explicit
// <tool_call> tag, or at a known tool name immediately preceding <arg_key>.
// start is where forwarded thinking must stop, body where the call text begins,
// and prefix a tool name that was already forwarded (so it precedes body).
func (r *thinkingScanner) toolStartInThinking(acc string) (start, body int, prefix string, ok bool) {
	if i := strings.Index(acc, glmToolOpenTag); i >= 0 {
		return i, i + len(glmToolOpenTag), "", true
	}
	i := strings.Index(acc, glmArgKeyOpen)
	if i < 0 {
		return 0, 0, "", false
	}
	before := r.recentTail + acc[:i]
	for _, t := range r.parser.tools {
		if !strings.HasSuffix(before, t.Name) {
			continue
		}
		if len(t.Name) <= i {
			// Whole name is inside acc: cut it out of the forwarded thinking.
			return i - len(t.Name), i - len(t.Name), "", true
		}
		// Name straddles already-forwarded text: keep it, supply it to the parser.
		return 0, i, t.Name, true
	}
	return 0, 0, "", false
}

// overlapSuffix reports how many trailing bytes of s form a proper prefix of tag.
func overlapSuffix(s, tag string) int {
	for n := len(tag) - 1; n > 0; n-- {
		if strings.HasSuffix(s, tag[:n]) {
			return n
		}
	}
	return 0
}

func (r *thinkingScanner) emit(out *feedResult, text string) {
	if text == "" {
		return
	}
	out.text += text
	r.remember(text)
}

func (r *thinkingScanner) feed(thinking string, done bool) feedResult {
	var out feedResult
	r.pending.WriteString(thinking)

	for {
		acc := r.pending.String()
		if acc == "" {
			break
		}
		if r.capturing {
			idx := strings.Index(acc, glmToolClose)
			if idx < 0 {
				if !done {
					out.withheld = true
					return out
				}
				if tc, ok := r.parser.parse(r.namePrefix+acc, "thinking"); ok {
					out.toolCalls = append(out.toolCalls, tc)
				} else {
					r.emit(&out, acc)
				}
				r.pending.Reset()
				r.capturing = false
				r.namePrefix = ""
				break
			}
			raw, rest := acc[:idx], acc[idx+len(glmToolClose):]
			r.pending.Reset()
			r.pending.WriteString(strings.TrimLeftFunc(rest, unicode.IsSpace))
			r.capturing = false
			if tc, ok := r.parser.parse(r.namePrefix+raw, "thinking"); ok {
				out.toolCalls = append(out.toolCalls, tc)
			} else {
				r.emit(&out, glmToolOpenTag+raw+glmToolClose)
			}
			r.namePrefix = ""
			continue
		}

		if start, body, prefix, ok := r.toolStartInThinking(acc); ok {
			r.emit(&out, acc[:start])
			r.pending.Reset()
			r.pending.WriteString(acc[body:])
			r.namePrefix = prefix
			r.capturing = true
			continue
		}
		if done {
			r.emit(&out, acc)
			r.pending.Reset()
			break
		}
		// Withhold only a trailing fragment that might grow into a tag.
		hold := overlapSuffix(acc, glmToolOpenTag)
		if h := overlapSuffix(acc, glmArgKeyOpen); h > hold {
			hold = h
		}
		if hold > 0 {
			r.emit(&out, acc[:len(acc)-hold])
			r.pending.Reset()
			r.pending.WriteString(acc[len(acc)-hold:])
			out.withheld = true
		} else {
			r.emit(&out, acc)
			r.pending.Reset()
		}
		break
	}
	return out
}

// --- Integration with the /api/chat response path ---------------------------

// glmChatChunk is the subset of an /api/chat chunk we need to inspect.
type glmChatChunk struct {
	root     map[string]json.RawMessage
	message  map[string]json.RawMessage
	content  string
	thinking string
	done     bool
	hasTC    bool
}

func parseGLMChatChunk(line []byte) (*glmChatChunk, bool) {
	c := &glmChatChunk{}
	if err := json.Unmarshal(line, &c.root); err != nil || c.root == nil {
		return nil, false
	}
	if rawDone, ok := c.root["done"]; ok {
		json.Unmarshal(rawDone, &c.done)
	}
	rawMsg, ok := c.root["message"]
	if !ok {
		return c, true
	}
	if err := json.Unmarshal(rawMsg, &c.message); err != nil || c.message == nil {
		return nil, false
	}
	if raw, ok := c.message["content"]; ok {
		json.Unmarshal(raw, &c.content)
	}
	if raw, ok := c.message["thinking"]; ok {
		json.Unmarshal(raw, &c.thinking)
	}
	if rawTC, ok := c.message["tool_calls"]; ok && !isNullOrEmptyJSONArray(rawTC) {
		c.hasTC = true
	}
	return c, true
}

// renderLine builds one non-final chunk line from the template chunk with the
// given content, thinking and optional tool call.
func (c *glmChatChunk) renderLine(content, thinking string, tc *recoveredToolCall) []byte {
	root := make(map[string]json.RawMessage, len(c.root))
	for k, v := range c.root {
		root[k] = v
	}
	root["done"] = json.RawMessage("false")
	for _, k := range []string{"done_reason", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"} {
		delete(root, k)
	}
	msg := make(map[string]json.RawMessage, len(c.message)+2)
	for k, v := range c.message {
		msg[k] = v
	}
	if _, ok := msg["role"]; !ok {
		msg["role"] = json.RawMessage(`"assistant"`)
	}
	contentJSON, _ := json.Marshal(content)
	msg["content"] = contentJSON
	delete(msg, "thinking")
	if thinking != "" {
		thinkingJSON, _ := json.Marshal(thinking)
		msg["thinking"] = thinkingJSON
	}
	delete(msg, "tool_calls")
	if tc != nil {
		tcJSON, _ := json.Marshal([]recoveredToolCall{*tc})
		msg["tool_calls"] = tcJSON
	}
	msgJSON, _ := json.Marshal(msg)
	root["message"] = msgJSON
	line, _ := json.Marshal(root)
	return line
}

// renderFinal re-emits the final (done) chunk with empty content/thinking so
// the client still receives stats and done_reason.
func (c *glmChatChunk) renderFinal() []byte {
	root := make(map[string]json.RawMessage, len(c.root))
	for k, v := range c.root {
		root[k] = v
	}
	msg := make(map[string]json.RawMessage, len(c.message))
	for k, v := range c.message {
		msg[k] = v
	}
	msg["content"] = json.RawMessage(`""`)
	delete(msg, "thinking")
	delete(msg, "tool_calls")
	msgJSON, _ := json.Marshal(msg)
	root["message"] = msgJSON
	line, _ := json.Marshal(root)
	return line
}

// glmStreamFilter rewrites a stream of /api/chat NDJSON lines. It returns the
// lines to forward for the given input line.
type glmStreamFilter struct {
	parser   *toolCallParser
	content  *contentScanner
	thinking *thinkingScanner

	sawContent  bool
	sawToolCall bool
	thinkingAcc strings.Builder // thinking forwarded so far, for the fallback
	endpoint    string

	loopKey     string // tool-call key that already repeats in the request history
	loopRepeats int
}

func newGLMStreamFilter(tools []requestToolDef, endpoint string) *glmStreamFilter {
	if len(tools) == 0 {
		return nil
	}
	p := &toolCallParser{tools: tools, endpoint: endpoint}
	return &glmStreamFilter{
		parser:   p,
		content:  &contentScanner{parser: p},
		thinking: &thinkingScanner{parser: p},
		endpoint: endpoint,
	}
}

func (f *glmStreamFilter) noteThinking(s string) {
	if f.thinkingAcc.Len() < maxThinkingKeep {
		f.thinkingAcc.WriteString(s)
	}
}

func (f *glmStreamFilter) process(line []byte) [][]byte {
	c, ok := parseGLMChatChunk(line)
	if !ok || c.message == nil {
		return [][]byte{line}
	}

	if c.hasTC {
		// A real tool call from Ollama: flush anything withheld first.
		out := f.flush(c)
		if tcs := ollamaToolCallsOf(c); len(tcs) == 1 {
			if notice, loop := f.loopNoticeFor(tcs[0].Function.Name, tcs[0].Function.Arguments); loop {
				f.sawContent = true
				out = append(out, c.renderLine(notice, "", nil))
				if c.done {
					out = append(out, c.renderFinal())
				}
				return out
			}
		}
		f.sawToolCall = true
		return append(out, line)
	}

	var thinkRes, contentRes feedResult
	changed := false
	if c.thinking != "" {
		thinkRes = f.thinking.feed(c.thinking, c.done)
		if thinkRes.withheld || thinkRes.text != c.thinking || len(thinkRes.toolCalls) > 0 {
			changed = true
		}
	}
	if c.content != "" {
		contentRes = f.content.feed(c.content, c.done)
		if contentRes.withheld || contentRes.text != c.content || len(contentRes.toolCalls) > 0 {
			changed = true
		}
	}
	f.noteThinking(thinkRes.text)
	if contentRes.text != "" {
		f.sawContent = true
	}

	var tail [][]byte
	if c.done {
		tail = f.flush(c)
		// The done chunk itself may carry tool calls we are about to emit; account for them.
		pendingCalls := len(thinkRes.toolCalls) + len(contentRes.toolCalls)
		if !f.sawContent && !f.sawToolCall && pendingCalls == 0 && contentRes.text == "" {
			if f.thinkingAcc.Len() > 0 {
				// Thinking-only answer: promote it to content so the client shows something.
				log.Printf("glm_thinking_promoted_to_content path=%s chars=%d", f.endpoint, f.thinkingAcc.Len())
				tail = append(tail, c.renderLine(strings.TrimSpace(f.thinkingAcc.String()), "", nil))
			} else {
				log.Printf("glm_empty_response_notice path=%s", f.endpoint)
				tail = append(tail, c.renderLine(emptyResponseNotice, "", nil))
			}
		}
		if len(tail) > 0 {
			changed = true
		}
	}

	if !changed {
		return [][]byte{line}
	}

	var out [][]byte
	if thinkRes.text != "" {
		out = append(out, c.renderLine("", thinkRes.text, nil))
	}
	out = append(out, f.renderToolCalls(c, thinkRes.toolCalls)...)
	if contentRes.text != "" {
		out = append(out, c.renderLine(contentRes.text, "", nil))
	}
	out = append(out, f.renderToolCalls(c, contentRes.toolCalls)...)
	out = append(out, tail...)
	if c.done {
		out = append(out, c.renderFinal())
	}
	return out
}

// flush resolves anything the scanners are still holding (before forwarding a
// real tool call, or at end of stream).
func (f *glmStreamFilter) flush(c *glmChatChunk) [][]byte {
	var out [][]byte
	if f.thinking.pending.Len() > 0 {
		res := f.thinking.feed("", true)
		if res.text != "" {
			f.noteThinking(res.text)
			out = append(out, c.renderLine("", res.text, nil))
		}
		out = append(out, f.renderToolCalls(c, res.toolCalls)...)
	}
	if f.content.pending.Len() > 0 {
		res := f.content.feed("", true)
		if res.text != "" {
			f.sawContent = true
			out = append(out, c.renderLine(res.text, "", nil))
		}
		out = append(out, f.renderToolCalls(c, res.toolCalls)...)
	}
	return out
}

// renderToolCalls emits recovered tool calls, replacing one that continues a
// detected loop with a notice.
func (f *glmStreamFilter) renderToolCalls(c *glmChatChunk, tcs []recoveredToolCall) [][]byte {
	var out [][]byte
	for i := range tcs {
		if notice, loop := f.loopNoticeFor(tcs[i].Function.Name, tcs[i].Function.Arguments); loop {
			f.sawContent = true
			out = append(out, c.renderLine(notice, "", nil))
			continue
		}
		f.sawToolCall = true
		out = append(out, c.renderLine("", "", &tcs[i]))
	}
	return out
}

// recoverGLMToolCallsInResponse repairs a non-streaming /api/chat response body
// by running it through the same filter as a single final chunk.
func recoverGLMToolCallsInResponse(body []byte, tools []requestToolDef, endpoint string, loopKey string, loopRepeats int) []byte {
	f := newGLMStreamFilter(tools, endpoint)
	if f == nil {
		return body
	}
	f.setLoopGuard(loopKey, loopRepeats)
	c, ok := parseGLMChatChunk(body)
	if !ok || c.message == nil || c.hasTC {
		return body
	}
	if !c.done {
		return body
	}
	lines := f.process(body)
	if len(lines) == 1 && string(lines[0]) == string(body) {
		return body
	}
	// Merge the emitted lines back into one response.
	var content, thinking strings.Builder
	var toolCalls []recoveredToolCall
	for _, l := range lines {
		lc, ok := parseGLMChatChunk(l)
		if !ok || lc.message == nil {
			continue
		}
		content.WriteString(lc.content)
		thinking.WriteString(lc.thinking)
		if raw, ok := lc.message["tool_calls"]; ok && !isNullOrEmptyJSONArray(raw) {
			var tcs []recoveredToolCall
			if json.Unmarshal(raw, &tcs) == nil {
				toolCalls = append(toolCalls, tcs...)
			}
		}
	}
	msg := make(map[string]json.RawMessage, len(c.message)+1)
	for k, v := range c.message {
		msg[k] = v
	}
	contentJSON, _ := json.Marshal(content.String())
	msg["content"] = contentJSON
	delete(msg, "thinking")
	if thinking.Len() > 0 {
		thinkingJSON, _ := json.Marshal(thinking.String())
		msg["thinking"] = thinkingJSON
	}
	delete(msg, "tool_calls")
	if len(toolCalls) > 0 {
		tcJSON, _ := json.Marshal(toolCalls)
		msg["tool_calls"] = tcJSON
	}
	msgJSON, err := json.Marshal(msg)
	if err != nil {
		return body
	}
	root := make(map[string]json.RawMessage, len(c.root))
	for k, v := range c.root {
		root[k] = v
	}
	root["message"] = msgJSON
	out, err := json.Marshal(root)
	if err != nil {
		return body
	}
	return out
}

// --- Safety nets: tool-call loops and empty responses ------------------------

const (
	loopBreakRepeats = 3 // identical consecutive calls already in history before we intervene
	proxyNoticeTag   = "[ollama-metrics-proxy]"
)

// toolCallKey identifies a tool call by name plus its string-valued arguments,
// so calls that differ only in numeric fields (line numbers creeping by one)
// still count as the same call. Path-like values are reduced to their last
// segment: a looping model was observed garbling a directory name
// ("new-repos-2025" -> "newpos-2025") while re-reading the same file.
func toolCallKey(name string, args map[string]any) string {
	var parts []string
	for k, v := range args {
		if s, ok := v.(string); ok {
			s = strings.ToLower(strings.TrimSpace(s))
			if i := strings.LastIndex(strings.TrimRight(s, "/"), "/"); i >= 0 {
				s = s[i+1:]
			}
			parts = append(parts, k+"="+s)
		}
	}
	// deterministic order
	for i := 1; i < len(parts); i++ {
		for j := i; j > 0 && parts[j] < parts[j-1]; j-- {
			parts[j], parts[j-1] = parts[j-1], parts[j]
		}
	}
	return name + "|" + strings.Join(parts, "|")
}

// detectRepeatedToolCall inspects the request history: if the most recent
// assistant turns are each a single tool call with the same key, it returns
// that key and how many times in a row it appears. Tool-result messages are
// skipped; any other assistant message ends the run.
func detectRepeatedToolCall(body []byte) (key string, count int) {
	var req struct {
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Function struct {
					Name      string         `json:"name"`
					Arguments map[string]any `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", 0
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		m := req.Messages[i]
		switch m.Role {
		case "tool":
			continue
		case "assistant":
			if len(m.ToolCalls) != 1 || strings.TrimSpace(m.Content) != "" {
				return key, count
			}
			k := toolCallKey(m.ToolCalls[0].Function.Name, m.ToolCalls[0].Function.Arguments)
			if key == "" {
				key = k
			} else if k != key {
				return key, count
			}
			count++
		default:
			return key, count
		}
	}
	return key, count
}

// loopNotice is the content emitted in place of a looping tool call.
func loopNotice(name string, repeats int) string {
	return proxyNoticeTag + " Stopped a tool-call loop: the model called `" + name +
		"` with the same arguments " + itoa(repeats+1) + " times in a row without making progress. " +
		"Give it a more specific instruction, or enable the terminal tool so it can run commands directly."
}

const emptyResponseNotice = proxyNoticeTag + " The model returned an empty response (no text, no tool call)."

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

// setLoopGuard arms the filter with the key of a tool call that already
// repeats in the request history.
func (f *glmStreamFilter) setLoopGuard(key string, repeats int) {
	if repeats >= loopBreakRepeats {
		f.loopKey = key
		f.loopRepeats = repeats
	}
}

// loopNoticeFor reports whether the given tool call continues the loop and, if
// so, the notice content that should replace it.
func (f *glmStreamFilter) loopNoticeFor(name string, args map[string]any) (string, bool) {
	if f.loopKey == "" || toolCallKey(name, args) != f.loopKey {
		return "", false
	}
	log.Printf("glm_toolcall_loop_broken path=%s tool=%s repeats=%d", f.endpoint, name, f.loopRepeats+1)
	return loopNotice(name, f.loopRepeats), true
}

// ollamaToolCallsOf extracts name/arguments from a chunk that carries native
// Ollama tool_calls.
func ollamaToolCallsOf(c *glmChatChunk) []recoveredToolCall {
	raw, ok := c.message["tool_calls"]
	if !ok {
		return nil
	}
	var tcs []recoveredToolCall
	if json.Unmarshal(raw, &tcs) != nil {
		return nil
	}
	return tcs
}
