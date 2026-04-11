package main

import (
	"flag"
	"log"
	"net/http"
	"net/url"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	listenAddr := flag.String("listen", ":11435", "Proxy listen address")
	metricsAddr := flag.String("metrics-listen", ":9836", "Metrics listen address")
	ollamaAddr := flag.String("ollama-url", "http://localhost:11434", "Ollama backend URL")
	flag.Parse()

	target, err := url.Parse(*ollamaAddr)
	if err != nil {
		log.Fatalf("invalid ollama URL: %v", err)
	}

	p := newProxy(target, prometheus.DefaultRegisterer)

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
