/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package router

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// modelsList GETs /v1/models through the harness and returns its data items.
func modelsList(t *testing.T, h *proxyHarness) []map[string]any {
	t.Helper()
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("/v1/models = %d, want 200", rec.Code)
	}
	var got struct {
		Object string           `json:"object"`
		Data   []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode /v1/models: %v", err)
	}
	if got.Object != "list" {
		t.Errorf("object = %q, want list", got.Object)
	}
	return got.Data
}

func entryByID(data []map[string]any, id string) map[string]any {
	for _, e := range data {
		if e["id"] == id {
			return e
		}
	}
	return nil
}

// A parked engine must still describe itself: what an operator declares is
// published without a single request reaching the backend.
func TestProxyModelsPublishDeclaredDetails(t *testing.T) {
	h := newProxyHarness(t)
	ctx := int32(393216)
	h.cfg.Backends[0].Capabilities = []string{"long-context", "vision", "tools"}
	h.cfg.Backends[0].ModelDetails = &ModelDetails{
		ContextLength:    &ctx,
		InputModalities:  []string{"text", "image"},
		OutputModalities: []string{"text"},
	}

	entry := entryByID(modelsList(t, h), "local-qwen")
	if entry == nil {
		t.Fatal("local-qwen missing from /v1/models")
	}
	caps, _ := entry["capabilities"].([]any)
	if len(caps) != 3 || caps[1] != "vision" {
		t.Errorf("capabilities = %v, want the three declared", entry["capabilities"])
	}
	meta, _ := entry["meta"].(map[string]any)
	if meta == nil || meta["n_ctx"] != float64(393216) {
		t.Errorf("meta = %v, want n_ctx 393216", entry["meta"])
	}
	arch, _ := entry["architecture"].(map[string]any)
	if arch == nil || len(arch["input_modalities"].([]any)) != 2 {
		t.Fatalf("architecture = %v, want two input modalities", entry["architecture"])
	}
	if arch["output_modalities"] == nil {
		t.Errorf("architecture = %v, want output_modalities", arch)
	}
}

// A backend with nothing declared is described from what it reported while it
// was up, so the fields are not simply missing for the operators who did not
// write them down.
func TestProxyModelsPublishDiscoveredDetails(t *testing.T) {
	h := newProxyHarness(t)
	cache := newDetailsCache()
	cache.store("local-qwen", discovered{
		ContextLength:   131072,
		InputModalities: []string{"text"},
		At:              cache.nowF(),
	})
	h.proxy.SetDetailsCache(cache)

	entry := entryByID(modelsList(t, h), "local-qwen")
	meta, _ := entry["meta"].(map[string]any)
	if meta == nil || meta["n_ctx"] != float64(131072) {
		t.Errorf("meta = %v, want the discovered n_ctx 131072", entry["meta"])
	}
	if _, ok := entry["capabilities"]; ok {
		t.Errorf("capabilities published for a backend that declares none: %v", entry["capabilities"])
	}
}

// What the operator declares outranks what an engine happened to report: a
// member may be reconfigured while parked, and the config is the intent.
func TestProxyModelsDeclaredWinOverDiscovered(t *testing.T) {
	h := newProxyHarness(t)
	ctx := int32(393216)
	h.cfg.Backends[0].ModelDetails = &ModelDetails{
		ContextLength:   &ctx,
		InputModalities: []string{"text", "image"},
	}
	cache := newDetailsCache()
	cache.store("local-qwen", discovered{
		ContextLength:   8192,
		InputModalities: []string{"text"},
		At:              cache.nowF(),
	})
	h.proxy.SetDetailsCache(cache)

	entry := entryByID(modelsList(t, h), "local-qwen")
	meta, _ := entry["meta"].(map[string]any)
	if meta["n_ctx"] != float64(393216) {
		t.Errorf("n_ctx = %v, want the declared 393216", meta["n_ctx"])
	}
	arch, _ := entry["architecture"].(map[string]any)
	if len(arch["input_modalities"].([]any)) != 2 {
		t.Errorf("input_modalities = %v, want the declared two", arch["input_modalities"])
	}
}

// A self-report stops being usable once the engine has been silent long enough
// that it could have been replaced by one configured differently.
func TestDiscoveredDetailsExpire(t *testing.T) {
	cache := newDetailsCache()
	cache.store("local-qwen", discovered{ContextLength: 4096, At: time.Now().Add(-detailsTTL - time.Minute)})
	if _, ok := cache.get("local-qwen"); ok {
		t.Error("a self-report older than the TTL is still being served")
	}
}

// strataListing is what a running Strata answers on /v1/models: the id, its
// alias, the context window under meta.n_ctx, and the modalities.
const strataListing = `{"object":"list","data":[{"id":"qwen3.8-flash-next-iq3_s","object":"model","status":{"value":"loaded"},"meta":{"n_ctx":393216},"architecture":{"input_modalities":["text","image"],"output_modalities":["text"]},"aliases":["qfn-alias"]},{"id":"qfn-alias","object":"model","alias_of":"qwen3.8-flash-next-iq3_s"}]}`

func TestDetailsScraperRecordsSelfReport(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(strataListing))
	}))
	defer engine.Close()

	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelDebug}))
	cfg := &Config{Backends: []Backend{{Name: "strata-qwen38", Tier: "local", Address: engine.URL}}}
	scraper, cache := NewDetailsScraper(cfg, NewDispatcher(cfg), logger)
	if !scraper.HasTargets() {
		t.Fatal("a local backend should be polled")
	}
	scraper.pollOnce(context.Background())

	d, ok := cache.get("strata-qwen38")
	if !ok {
		t.Fatal("nothing recorded from a running engine")
	}
	if d.ContextLength != 393216 {
		t.Errorf("context length = %d, want 393216 from meta.n_ctx", d.ContextLength)
	}
	if len(d.InputModalities) != 2 || d.InputModalities[1] != "image" {
		t.Errorf("input modalities = %v, want text and image", d.InputModalities)
	}
}

// vLLM spells the context window differently; the poller reads both.
func TestDetailsScraperReadsVLLMShape(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"svc","object":"model","max_model_len":32768}]}`))
	}))
	defer engine.Close()

	cfg := &Config{Backends: []Backend{{Name: "svc", Tier: "local", Address: engine.URL, ServedModel: "svc"}}}
	scraper, cache := NewDetailsScraper(cfg, NewDispatcher(cfg), slog.Default())
	scraper.pollOnce(context.Background())

	d, ok := cache.get("svc")
	if !ok || d.ContextLength != 32768 {
		t.Errorf("discovered = %+v ok=%v, want max_model_len 32768", d, ok)
	}
}

// A parked member refuses the connection. That is the normal state of a pool
// member and must not look like a description of anything.
func TestDetailsScraperRecordsNothingForParkedBackend(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	address := engine.URL
	engine.Close()

	cfg := &Config{Backends: []Backend{{Name: "parked", Tier: "local", Address: address}}}
	scraper, cache := NewDetailsScraper(cfg, NewDispatcher(cfg), slog.Default())
	scraper.pollOnce(context.Background())

	if _, ok := cache.get("parked"); ok {
		t.Error("a backend that refused the connection got a self-report anyway")
	}
}

// Cloud providers list opaque ids and rarely report a window or modalities;
// polling them would spend provider requests for nothing.
func TestDetailsScraperSkipsCloudBackends(t *testing.T) {
	cfg := &Config{Backends: []Backend{
		{Name: "cloud", Tier: "cloud", Address: "https://api.example.com", Provider: "anthropic", Model: "m"},
	}}
	scraper, _ := NewDetailsScraper(cfg, NewDispatcher(cfg), slog.Default())
	if scraper.HasTargets() {
		t.Error("a cloud backend was queued for a details poll")
	}
}

func TestPickDetailsEntry(t *testing.T) {
	data := []detailsEntry{
		{ID: "alias-a", AliasOf: "served"},
		{ID: "served", Meta: struct {
			NCtx int32 `json:"n_ctx"`
		}{NCtx: 4096}},
		{ID: "other", MaxModelLen: 1},
	}
	b := &Backend{Name: "display", DisplayName: "client-facing", ServedModel: "served"}
	if got := pickDetailsEntry(data, b); got == nil || got.ID != "served" {
		t.Errorf("picked %+v, want the entry the backend serves upstream", got)
	}

	// No id matches: the first primary entry describes the backend, never an
	// aliased duplicate.
	b2 := &Backend{Name: "unrelated"}
	if got := pickDetailsEntry(data, b2); got == nil || got.ID != "served" {
		t.Errorf("picked %+v, want the first primary entry", got)
	}

	if got := pickDetailsEntry(nil, b2); got != nil {
		t.Errorf("picked %+v from an empty listing, want nil", got)
	}
}
