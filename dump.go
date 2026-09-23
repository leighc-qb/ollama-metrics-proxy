package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// Debug body dumps for /api/chat.
//
// When the configured dump directory exists, every /api/chat request body and
// the full upstream response (all streamed chunks, or the single JSON body) are
// written to <dir>/<timestamp>-<seq>.req.json and .resp.ndjson. Dumping is
// toggled purely by creating or removing the directory, so it can be switched
// on and off without restarting the service or touching its unit file.
// Dumps contain prompts and model output; treat them as sensitive.

var dumpSeq uint64

type bodyDump struct {
	resp *os.File
}

// newBodyDump starts a dump for one request, or returns nil when dumping is
// off (no directory configured, or the directory does not exist).
func (p *proxy) newBodyDump(endpoint string, body []byte) *bodyDump {
	if p.dumpDir == "" {
		return nil
	}
	st, err := os.Stat(p.dumpDir)
	if err != nil || !st.IsDir() {
		return nil
	}
	seq := atomic.AddUint64(&dumpSeq, 1)
	base := filepath.Join(p.dumpDir, fmt.Sprintf("%s-%04d", time.Now().Format("20060102-150405.000"), seq))
	if err := os.WriteFile(base+".req.json", body, 0o600); err != nil {
		log.Printf("body_dump_failed path=%s err=%v", endpoint, err)
		return nil
	}
	f, err := os.OpenFile(base+".resp.ndjson", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		log.Printf("body_dump_failed path=%s err=%v", endpoint, err)
		return nil
	}
	log.Printf("body_dump path=%s file=%s", endpoint, base)
	return &bodyDump{resp: f}
}

func (d *bodyDump) Write(b []byte) (int, error) {
	return d.resp.Write(b)
}

func (d *bodyDump) Close() {
	if d != nil && d.resp != nil {
		d.resp.Close()
	}
}
