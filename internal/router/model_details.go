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
	"sync"
	"time"
)

// ModelDetails is the declared description of a backend's model, mirrored from
// api/v1alpha1.ModelDetails by the controller's router config builder. It is
// what a client needs to plan around a model: how much context it holds, and
// which modalities it accepts and produces.
type ModelDetails struct {
	ContextLength    *int32   `json:"contextLength,omitempty"`
	InputModalities  []string `json:"inputModalities,omitempty"`
	OutputModalities []string `json:"outputModalities,omitempty"`
}

const (
	// detailsInterval is how often the details scraper asks a local backend
	// what it is. It is a background convenience, not a hot path: a declared
	// modelDetails needs no polling at all, and a discovered value is usually
	// stable for the life of the pod.
	detailsInterval = 60 * time.Second
	// detailsTimeout bounds one /v1/models read. A parked engine refuses the
	// connection; the poll must not linger behind it.
	detailsTimeout = 2 * time.Second
	// detailsTTL is how long a discovered value stays usable after the backend
	// last answered. Beyond it the field goes unset again rather than being
	// reported as current, because an engine that stopped answering may have
	// been replaced by one configured differently.
	detailsTTL = 30 * time.Minute
)

// discovered is what a running backend reported about itself.
type discovered struct {
	ContextLength    int32
	InputModalities  []string
	OutputModalities []string
	At               time.Time
}

// detailsCache holds the last self-report of each backend, keyed by backend
// name. Written only by the scraper, read by the model list handler.
type detailsCache struct {
	mu   sync.RWMutex
	by   map[string]discovered
	nowF func() time.Time
}

func newDetailsCache() *detailsCache {
	return &detailsCache{by: map[string]discovered{}, nowF: time.Now}
}

func (c *detailsCache) store(backend string, d discovered) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.by[backend] = d
}

// get returns a backend's self-report if it is still inside detailsTTL.
func (c *detailsCache) get(backend string) (discovered, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	d, ok := c.by[backend]
	if !ok {
		return discovered{}, false
	}
	if c.nowF().Sub(d.At) > detailsTTL {
		return discovered{}, false
	}
	return d, true
}

// DetailsScraper asks each local backend what it serves, so a backend whose
// modelDetails were never declared still describes itself on /v1/models for as
// long as it is up. It changes nothing about routing, and a backend that is
// parked simply has nothing to report.
type DetailsScraper struct {
	disp     *Dispatcher
	client   *http.Client
	backends []*Backend
	cache    *detailsCache
	logger   *slog.Logger
}

// NewDetailsScraper indexes the backends worth asking: local ones the proxy can
// address. Cloud providers list opaque ids and rarely report a context window
// or modalities, so polling them would cost requests for nothing. The cache it
// returns is the one the model list handler reads.
func NewDetailsScraper(cfg *Config, disp *Dispatcher, logger *slog.Logger) (*DetailsScraper, *detailsCache) {
	cache := newDetailsCache()
	s := &DetailsScraper{
		disp:   disp,
		cache:  cache,
		client: &http.Client{Timeout: detailsTimeout},
		logger: logger,
	}
	for i := range cfg.Backends {
		b := &cfg.Backends[i]
		if b.Tier == "local" && b.Address != "" {
			s.backends = append(s.backends, b)
		}
	}
	return s, cache
}

// HasTargets reports whether any backend can be asked.
func (s *DetailsScraper) HasTargets() bool { return len(s.backends) > 0 }

// Start polls every detailsInterval until ctx ends.
func (s *DetailsScraper) Start(ctx context.Context) {
	t := time.NewTicker(detailsInterval)
	defer t.Stop()
	for {
		s.pollOnce(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (s *DetailsScraper) pollOnce(ctx context.Context) {
	for _, b := range s.backends {
		if d, ok := s.poll(ctx, b); ok {
			s.cache.store(b.Name, d)
		}
	}
}

// poll reads the backend's own /v1/models and takes the fields clients care
// about.
func (s *DetailsScraper) poll(ctx context.Context, b *Backend) (discovered, bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.Address+"/v1/models", nil)
	if err != nil {
		return discovered{}, false
	}
	if err := s.disp.applyCredentials(b, req); err != nil {
		return discovered{}, false
	}
	resp, err := s.client.Do(req)
	if err != nil {
		s.logger.Debug("model details poll failed", "backend", b.Name, "error", err)
		return discovered{}, false
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		s.logger.Debug("model details poll not usable", "backend", b.Name, "status", resp.StatusCode)
		return discovered{}, false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return discovered{}, false
	}
	var payload struct {
		Data []detailsEntry `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		s.logger.Debug("model details poll unreadable", "backend", b.Name, "error", err)
		return discovered{}, false
	}
	entry := pickDetailsEntry(payload.Data, b)
	if entry == nil {
		return discovered{}, false
	}
	d := discovered{At: s.cache.nowF()}
	if n := entry.MaxModelLen; n > 0 {
		d.ContextLength = n
	} else if entry.Meta.NCtx > 0 {
		d.ContextLength = entry.Meta.NCtx
	}
	d.InputModalities = append([]string(nil), entry.Architecture.InputModalities...)
	d.OutputModalities = append([]string(nil), entry.Architecture.OutputModalities...)
	if d.ContextLength == 0 && len(d.InputModalities) == 0 && len(d.OutputModalities) == 0 {
		return discovered{}, false
	}
	return d, true
}

// detailsEntry is one item of a backend's /v1/models listing, as far as the
// details poller cares. The spellings differ per engine: llama.cpp and Strata
// report meta.n_ctx, vLLM reports max_model_len, and the architecture block is
// the OpenAI one.
type detailsEntry struct {
	ID          string `json:"id"`
	AliasOf     string `json:"alias_of,omitempty"`
	MaxModelLen int32  `json:"max_model_len"`
	Meta        struct {
		NCtx int32 `json:"n_ctx"`
	} `json:"meta"`
	Architecture struct {
		InputModalities  []string `json:"input_modalities"`
		OutputModalities []string `json:"output_modalities"`
	} `json:"architecture"`
}

// pickDetailsEntry chooses which item of a multi-model listing describes this
// backend: the id it expects upstream, then its own name, then its client-facing
// name, and failing all three the first primary entry - the single-item listing
// most engines return needs none of the matching. An aliased duplicate
// (alias_of set) never wins, so the primary entry's numbers are what gets
// recorded.
func pickDetailsEntry(data []detailsEntry, b *Backend) *detailsEntry {
	for _, want := range []string{b.outboundModel(), b.Name, b.DisplayName} {
		if want == "" {
			continue
		}
		for i := range data {
			if data[i].ID == want && data[i].AliasOf == "" {
				return &data[i]
			}
		}
	}
	for i := range data {
		if data[i].AliasOf == "" {
			return &data[i]
		}
	}
	if len(data) > 0 {
		return &data[0]
	}
	return nil
}

// modelEntry is one item of the proxy's own GET /v1/models answer. The four
// OpenAI fields are what the proxy has always published; the rest is what lets
// a client decide how much context to keep and whether a picture may be sent
// at all.
type modelEntry struct {
	ID           string         `json:"id"`
	Object       string         `json:"object"`
	Created      int64          `json:"created"`
	OwnedBy      string         `json:"owned_by"`
	Capabilities []string       `json:"capabilities,omitempty"`
	Meta         map[string]int `json:"meta,omitempty"`
	Architecture map[string]any `json:"architecture,omitempty"`
}

// buildModelEntry describes one backend for the model list. Declared
// modelDetails are authoritative and a discovered self-report fills only what
// they leave unset, so a parked engine is described exactly as its config says
// and a running one can still report something the operator never wrote down.
func buildModelEntry(b *Backend, id, ownedBy string, now int64, live discovered, hasLive bool) modelEntry {
	e := modelEntry{ID: id, Object: "model", Created: now, OwnedBy: ownedBy}
	if len(b.Capabilities) > 0 {
		e.Capabilities = append([]string(nil), b.Capabilities...)
	}
	var ctxLen int
	var inputs, outputs []string
	if b.ModelDetails != nil {
		if b.ModelDetails.ContextLength != nil {
			ctxLen = int(*b.ModelDetails.ContextLength)
		}
		inputs = append(inputs, b.ModelDetails.InputModalities...)
		outputs = append(outputs, b.ModelDetails.OutputModalities...)
	}
	if hasLive {
		if ctxLen == 0 && live.ContextLength > 0 {
			ctxLen = int(live.ContextLength)
		}
		if len(inputs) == 0 {
			inputs = append(inputs, live.InputModalities...)
		}
		if len(outputs) == 0 {
			outputs = append(outputs, live.OutputModalities...)
		}
	}
	if ctxLen > 0 {
		e.Meta = map[string]int{"n_ctx": ctxLen}
	}
	if len(inputs) > 0 || len(outputs) > 0 {
		e.Architecture = map[string]any{}
		if len(inputs) > 0 {
			e.Architecture["input_modalities"] = inputs
		}
		if len(outputs) > 0 {
			e.Architecture["output_modalities"] = outputs
		}
	}
	return e
}
