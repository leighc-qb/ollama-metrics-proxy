package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

// A naive implementation that unmarshals into map[string]any and re-marshals
// will silently corrupt integers larger than 2^53, because every JSON number
// becomes a float64. repairEmptyAssistantMessages must not do that: it works on
// json.RawMessage, so fields it does not touch keep their exact original bytes.
//
// This is a regression test for a real bug -- an earlier rewrite of this filter
// used map[string]any and turned seed 9007199254740993 into 9007199254740992.
func TestRepairPreservesLargeNumbersExactly(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{
			name: "large integer seed survives",
			body: `{"model":"m","seed":12345678901234567890,"messages":[{"role":"assistant","content":null}]}`,
			want: "12345678901234567890",
		},
		{
			name: "integer just above 2^53 survives",
			body: `{"model":"m","options":{"seed":9007199254740993},"messages":[{"role":"assistant","content":null}]}`,
			want: "9007199254740993",
		},
		{
			name: "high precision float survives",
			body: `{"model":"m","temperature":0.30000000000000004,"messages":[{"role":"assistant","content":null}]}`,
			want: "0.30000000000000004",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, repaired := repairEmptyAssistantMessages([]byte(tc.body))
			if repaired != 1 {
				t.Fatalf("repaired = %d, want 1", repaired)
			}
			if !bytes.Contains(got, []byte(tc.want)) {
				t.Errorf("numeric value was corrupted.\n  want substring: %s\n  got body:       %s", tc.want, got)
			}
		})
	}
}

// Fields the filter does not care about must survive untouched, including
// nested objects and unknown keys, since the body is forwarded to Ollama.
func TestRepairPreservesUnrelatedFields(t *testing.T) {
	body := []byte(`{"model":"m","stream":true,"options":{"num_ctx":262144,"num_gpu":99},` +
		`"keep_alive":"30m","some_future_field":{"a":[1,2,3]},` +
		`"messages":[{"role":"assistant","content":null}]}`)

	got, repaired := repairEmptyAssistantMessages(body)
	if repaired != 1 {
		t.Fatalf("repaired = %d, want 1", repaired)
	}

	var out map[string]any
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("repaired body is not valid JSON: %v", err)
	}
	for _, k := range []string{"model", "stream", "options", "keep_alive", "some_future_field"} {
		if _, ok := out[k]; !ok {
			t.Errorf("field %q was dropped: %s", k, got)
		}
	}
	if !bytes.Contains(got, []byte(`"num_ctx":262144`)) {
		t.Errorf("nested option changed representation: %s", got)
	}
}

// A message that carries real tool_calls legitimately uses content: null.
// Rewriting it would corrupt tool calling, so the whole body must come back
// byte-for-byte identical.
func TestRepairLeavesToolCallMessagesByteIdentical(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"assistant","content":null,` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"ls","arguments":"{}"}}]}]}`)

	got, repaired := repairEmptyAssistantMessages(body)
	if repaired != 0 {
		t.Fatalf("repaired = %d, want 0 -- tool_calls message must not be touched", repaired)
	}
	if !bytes.Equal(got, body) {
		t.Errorf("body was modified.\n  got:  %s\n  want: %s", got, body)
	}
}

// Nothing to repair means the body is forwarded completely unchanged, so the
// filter is a no-op for every non-Copilot client.
func TestRepairIsByteIdenticalNoOpWhenNothingToFix(t *testing.T) {
	bodies := []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"hello"}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":""}]}`,
		`{"model":"m","messages":[{"role":"user","content":null}]}`,
		`{"model":"m","prompt":"no messages array"}`,
		`not json at all`,
	}
	for _, b := range bodies {
		got, repaired := repairEmptyAssistantMessages([]byte(b))
		if repaired != 0 {
			t.Errorf("repaired = %d, want 0 for %s", repaired, b)
		}
		if !bytes.Equal(got, []byte(b)) {
			t.Errorf("body changed when it should not have.\n  in:  %s\n  out: %s", b, got)
		}
	}
}
