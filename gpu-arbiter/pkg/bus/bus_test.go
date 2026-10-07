package bus

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSubjects(t *testing.T) {
	if got := CommandSubject("engines", "fake", "load"); got != "devai.engines.fake.load" {
		t.Errorf("CommandSubject = %q", got)
	}
	if got := EventSubject("operator", "actions"); got != "devai.events.operator.actions" {
		t.Errorf("EventSubject = %q", got)
	}
	if got := ServiceSubject("laya-trainer", "trainer"); got != "devai.laya-trainer.trainer" {
		t.Errorf("ServiceSubject = %q", got)
	}
}

func TestParseCommandSubject(t *testing.T) {
	cases := []struct {
		in      string
		c, s, v string
		ok      bool
	}{
		{"devai.engines.vllm.load", "engines", "vllm", "load", true},
		{"devai.laya-trainer.trainer.cancel", "laya-trainer", "trainer", "cancel", true},
		{"devai.events.engines.vllm", "", "", "", false},
		{"devai.engines.vllm", "", "", "", false},
		{"other.engines.vllm.load", "", "", "", false},
		{"devai.engines.VLLM.load", "", "", "", false},
		{"devai.engines.*.load", "", "", "", false},
	}
	for _, tc := range cases {
		c, s, v, err := ParseCommandSubject(tc.in)
		if (err == nil) != tc.ok {
			t.Errorf("%q: err=%v, want ok=%v", tc.in, err, tc.ok)
			continue
		}
		if tc.ok && (c != tc.c || s != tc.s || v != tc.v) {
			t.Errorf("%q = %q %q %q", tc.in, c, s, v)
		}
	}
	if _, _, err := ParseServiceSubject("devai.operator.actions"); err != nil {
		t.Errorf("ParseServiceSubject: %v", err)
	}
	if _, _, err := ParseServiceSubject("devai.events.operator"); err == nil {
		t.Error("an event subject parsed as a service subject")
	}
}

func TestDecodeRequestEnvelope(t *testing.T) {
	cases := []struct {
		name string
		body string
		ok   bool
	}{
		{"minimal", `{"v":1,"job":0}`, true},
		{"load", `{"v":1,"job":7,"launch":{"backend":"fake"},"timeout_s":30}`, true},
		{"missing v", `{"job":7}`, false},
		{"wrong v", `{"v":2,"job":7}`, false},
		{"missing job", `{"v":1}`, false},
		{"negative job", `{"v":1,"job":-1}`, false},
		{"unknown field", `{"v":1,"job":1,"timeout":30}`, false},
		{"trailing", `{"v":1,"job":1}{}`, false},
		{"not json", `load please`, false},
	}
	for _, tc := range cases {
		_, err := DecodeRequest([]byte(tc.body))
		if (err == nil) != tc.ok {
			t.Errorf("%s: err=%v, want ok=%v", tc.name, err, tc.ok)
		}
	}
}

func TestReplyShapes(t *testing.T) {
	job := int64(3)
	b, err := Marshal(OKReply(State{Container: "engines", Service: "fake", Status: StatusServing, Job: &job}))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m["v"].(float64) != 1 || m["ok"] != true {
		t.Errorf("ok reply = %s", b)
	}
	st := m["state"].(map[string]any)
	for _, k := range []string{"container", "service", "status", "job", "launch", "since", "build", "sleeper", "error"} {
		if _, ok := st[k]; !ok {
			t.Errorf("state lacks %q: %s", k, b)
		}
	}

	b, _ = Marshal(ErrReply(CodeBusy, "another job's model is loaded", ""))
	r, err := DecodeReply(b)
	if err != nil {
		t.Fatal(err)
	}
	if r.OK || r.Error.Code != CodeBusy {
		t.Errorf("error reply = %s", b)
	}
	if !strings.Contains(string(b), `"log_tail":""`) {
		t.Errorf("error reply lacks log_tail: %s", b)
	}
	if _, err := DecodeReply([]byte(`{"v":1,"ok":false}`)); err == nil {
		t.Error("a refusal without an error decoded")
	}
}

func TestTrimLogTailKeepsTheEnd(t *testing.T) {
	long := strings.Repeat("early line\n", 10000) + "the failure\n"
	got := TrimLogTail(long)
	if len(got) > MaxLogTailBytes+16 {
		t.Errorf("trimmed tail is %d bytes", len(got))
	}
	if !strings.HasSuffix(got, "the failure\n") {
		t.Error("the end of the log was cut")
	}
	b, err := Marshal(ErrReply(CodeLoadFailed, "x", long))
	if err != nil {
		t.Fatalf("a reply with a trimmed tail does not fit the bus: %v", err)
	}
	if len(b) > MaxMessageBytes {
		t.Errorf("reply is %d bytes", len(b))
	}
}

// The bound is on the encoded size: a tail of control characters (six
// bytes each in JSON) and of <, >, & must still fit a reply.
func TestTrimLogTailBoundsTheEncodedSize(t *testing.T) {
	for _, chunk := range []string{"\x1b[0m\x01\x02\n", "<a href=x>&amp;</a>\n", "\u00e9\u4e2d\n"} {
		long := strings.Repeat(chunk, 40000) + "the failure\n"
		got := TrimLogTail(long)
		if !strings.HasSuffix(got, "the failure\n") {
			t.Errorf("%q: the end was cut", chunk)
		}
		b, err := Marshal(ErrReply(CodeLoadFailed, "load failed", long))
		if err != nil {
			t.Errorf("%q: the reply does not fit: %v", chunk, err)
			continue
		}
		if len(b) > MaxMessageBytes {
			t.Errorf("%q: reply of %d bytes", chunk, len(b))
		}
		if !utf8.ValidString(got) {
			t.Errorf("%q: the trimmed tail is not valid UTF-8", chunk)
		}
	}
}

func TestValidReason(t *testing.T) {
	for _, r := range []string{"scancel", "time_limit", "epilog", "operator"} {
		if !ValidReason(r) {
			t.Errorf("%q refused", r)
		}
	}
	for _, r := range []string{"", "Time", "a b", strings.Repeat("a", 33), "x-y"} {
		if ValidReason(r) {
			t.Errorf("%q accepted", r)
		}
	}
}
