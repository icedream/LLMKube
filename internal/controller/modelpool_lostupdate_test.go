/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package controller

import (
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	inferencev1alpha1 "github.com/defilantech/llmkube/api/v1alpha1"
)

// patchRecorder captures the wire form of every patch the reconciler sends.
type patchRecorder struct {
	client.Client
	patches []string
}

func (p *patchRecorder) Patch(ctx context.Context, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
	if data, err := patch.Data(obj); err == nil {
		p.patches = append(p.patches, string(data))
	}
	return p.Client.Patch(ctx, obj, patch, opts...)
}

// TestEnsureReplicasPatchIsGuardedByResourceVersion covers the lost update behind
// a pool swap that never starts. spec.replicas has two writers: this reconciler,
// holding members at zero or one, and the router's activator, scaling a member up
// for a request. The reconciler patches a client read from the top of its
// reconcile, so an activation that lands in between is only preserved when the
// patch carries a resourceVersion and the API server answers with a conflict.
func TestEnsureReplicasPatchIsGuardedByResourceVersion(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := inferencev1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("scheme: %v", err)
	}
	member := &inferencev1alpha1.InferenceService{
		ObjectMeta: metav1.ObjectMeta{Name: "member-a", Namespace: "llmpool"},
		Spec:       inferencev1alpha1.InferenceServiceSpec{Replicas: ptr.To(int32(0))},
	}
	recorder := &patchRecorder{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(member).Build()}
	r := &ModelPoolReconciler{Client: recorder}

	target := &inferencev1alpha1.InferenceService{}
	if err := recorder.Get(context.Background(), client.ObjectKey{Name: "member-a", Namespace: "llmpool"}, target); err != nil {
		t.Fatalf("get member: %v", err)
	}
	target.Spec.Replicas = ptr.To(int32(1))

	if err := r.ensureReplicas(context.Background(), target, 0); err != nil {
		t.Fatalf("drain patch: %v", err)
	}
	if len(recorder.patches) != 1 {
		t.Fatalf("want one patch, got %d", len(recorder.patches))
	}
	if !strings.Contains(recorder.patches[0], `"resourceVersion"`) {
		t.Fatalf("drain patch carries no resourceVersion, so a concurrent activation would be erased: %s",
			recorder.patches[0])
	}
}
