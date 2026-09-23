package main

import (
	"flag"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	listenAddr := flag.String("listen", ":11435", "Proxy listen address")
	metricsAddr := flag.String("metrics-listen", ":9836", "Metrics listen address")
	ollamaAddr := flag.String("ollama-url", "http://localhost:11434", "Ollama backend URL")
	dumpDir := flag.String("dump-dir", defaultDumpDir(), "Directory for /api/chat request/response dumps; dumping is active only while the directory exists")
	flag.Parse()

	target, err := url.Parse(*ollamaAddr)
	if err != nil {
		log.Fatalf("invalid ollama URL: %v", err)
	}

	p := newProxy(target, prometheus.DefaultRegisterer)
	p.dumpDir = *dumpDir
	log.Printf("body dumps of /api/chat are written to %s while that directory exists", *dumpDir)

	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		log.Printf("metrics listening on %s", *metricsAddr)
		log.Fatal(http.ListenAndServe(*metricsAddr, mux))
	}()

	server := &http.Server{
		Addr:              *listenAddr,
		Handler:           p,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("proxy listening on %s -> %s", *listenAddr, *ollamaAddr)
	log.Fatal(server.ListenAndServe())
}

// defaultDumpDir is ~/.cache/ollama-metrics-proxy/dump, or empty (dumping
// unavailable) when the home directory cannot be determined.
func defaultDumpDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".cache", "ollama-metrics-proxy", "dump")
}
