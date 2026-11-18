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
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	inferencev1alpha1 "github.com/defilantech/llmkube/api/v1alpha1"
)

func mpoolFor(resident, pending, def string, names ...string) *inferencev1alpha1.ModelPool {
	members := make([]inferencev1alpha1.ModelPoolMember, 0, len(names))
	for _, n := range names {
		members = append(members, inferencev1alpha1.ModelPoolMember{
			InferenceServiceRef: corev1.LocalObjectReference{Name: n},
		})
	}
	return &inferencev1alpha1.ModelPool{
		Spec:   inferencev1alpha1.ModelPoolSpec{Members: members, Default: def},
		Status: inferencev1alpha1.ModelPoolStatus{ResidentMember: resident, PendingMember: pending},
	}
}

func isvcReplicas(r int32) *inferencev1alpha1.InferenceService {
	return &inferencev1alpha1.InferenceService{Spec: inferencev1alpha1.InferenceServiceSpec{Replicas: ptr.To(r)}}
}

func membersMap(scaled map[string]int32) map[string]*inferencev1alpha1.InferenceService {
	m := make(map[string]*inferencev1alpha1.InferenceService, len(scaled))
	for name, r := range scaled {
		m[name] = isvcReplicas(r)
	}
	return m
}

// membersSuspended builds the member set of membersMap and parks the named
// members with spec.suspend. A member listed only in suspended is present with
// zero replicas.
func membersSuspended(scaled map[string]int32, suspended []string) map[string]*inferencev1alpha1.InferenceService {
	m := membersMap(scaled)
	for _, name := range suspended {
		isvc, ok := m[name]
		if !ok {
			isvc = isvcReplicas(0)
			m[name] = isvc
		}
		isvc.Spec.Suspend = true
	}
	return m
}

func TestResolveSlotOwner(t *testing.T) {
	tests := []struct {
		name      string
		pool      *inferencev1alpha1.ModelPool
		members   map[string]int32
		suspended []string
		want      string
	}{
		{
			name:    "cold pool, nothing scaled, spec.default warms",
			pool:    mpoolFor("", "", "coder", "coder", "judge"),
			members: map[string]int32{"coder": 0, "judge": 0},
			want:    "coder",
		},
		{
			name:    "cold pool, nothing scaled, no default stays cold",
			pool:    mpoolFor("", "", "", "coder", "judge"),
			members: map[string]int32{"coder": 0, "judge": 0},
			want:    "",
		},
		{
			name:    "cold pool, scaled member matching default wins",
			pool:    mpoolFor("", "", "judge", "coder", "judge"),
			members: map[string]int32{"coder": 1, "judge": 1},
			want:    "judge",
		},
		{
			name:    "cold pool, scaled members, no default takes declaration order",
			pool:    mpoolFor("", "", "", "coder", "judge"),
			members: map[string]int32{"coder": 1, "judge": 1},
			want:    "coder",
		},
		{
			name:    "cold pool, pending swap continues first",
			pool:    mpoolFor("", "judge", "coder", "coder", "judge"),
			members: map[string]int32{"coder": 0, "judge": 0},
			want:    "judge",
		},
		{
			name:    "warm pool, fresh activation supersedes resident",
			pool:    mpoolFor("coder", "", "", "coder", "judge"),
			members: map[string]int32{"coder": 1, "judge": 1},
			want:    "judge",
		},
		{
			name:    "warm pool, pending among newly activated wins",
			pool:    mpoolFor("coder", "gemma", "", "coder", "judge", "gemma"),
			members: map[string]int32{"coder": 1, "judge": 1, "gemma": 1},
			want:    "gemma",
		},
		{
			name:    "warm pool, no fresh activation, pending swap continues",
			pool:    mpoolFor("coder", "judge", "", "coder", "judge"),
			members: map[string]int32{"coder": 1, "judge": 0},
			want:    "judge",
		},
		{
			name:    "warm pool, quiescent keeps resident",
			pool:    mpoolFor("coder", "", "", "coder", "judge"),
			members: map[string]int32{"coder": 1, "judge": 0},
			want:    "coder",
		},
		{
			name:      "warm pool, suspended activation does not displace resident",
			pool:      mpoolFor("coder", "", "", "coder", "judge"),
			members:   map[string]int32{"coder": 1, "judge": 1},
			suspended: []string{"judge"},
			want:      "coder",
		},
		{
			name:      "warm pool, suspended pending swap is ignored",
			pool:      mpoolFor("coder", "judge", "", "coder", "judge"),
			members:   map[string]int32{"coder": 1, "judge": 0},
			suspended: []string{"judge"},
			want:      "coder",
		},
		{
			name:      "warm pool, suspended resident falls back to default",
			pool:      mpoolFor("coder", "", "judge", "coder", "judge"),
			members:   map[string]int32{"coder": 0, "judge": 0},
			suspended: []string{"coder"},
			want:      "judge",
		},
		{
			name:      "cold pool, suspended default stays cold",
			pool:      mpoolFor("", "", "coder", "coder", "judge"),
			members:   map[string]int32{"coder": 0, "judge": 0},
			suspended: []string{"coder"},
			want:      "",
		},
		{
			name:      "cold pool, suspended pending does not block a fresh activation",
			pool:      mpoolFor("", "coder", "", "coder", "judge"),
			members:   map[string]int32{"coder": 0, "judge": 1},
			suspended: []string{"coder"},
			want:      "judge",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveSlotOwner(tc.pool, membersSuspended(tc.members, tc.suspended))
			if got != tc.want {
				t.Errorf("resolveSlotOwner = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestReplicasOf(t *testing.T) {
	if got := replicasOf(&inferencev1alpha1.InferenceService{}); got != 1 {
		t.Errorf("replicasOf(nil replicas) = %d, want 1 (apiserver default)", got)
	}
	if got := replicasOf(isvcReplicas(0)); got != 0 {
		t.Errorf("replicasOf(0) = %d, want 0", got)
	}
	if got := replicasOf(isvcReplicas(3)); got != 3 {
		t.Errorf("replicasOf(3) = %d, want 3", got)
	}
}

func TestMemberReady(t *testing.T) {
	if memberReady(nil) {
		t.Error("memberReady(nil) = true, want false")
	}
	notReady := &inferencev1alpha1.InferenceService{}
	notReady.Status.Phase = "Pending"
	if memberReady(notReady) {
		t.Error("memberReady(Pending) = true, want false")
	}
	ready := &inferencev1alpha1.InferenceService{}
	ready.Status.Phase = PhaseReady
	if !memberReady(ready) {
		t.Error("memberReady(Ready) = false, want true")
	}
}

func TestMemberSuspended(t *testing.T) {
	if memberSuspended(nil) {
		t.Error("memberSuspended(nil) = true, want false")
	}
	suspended := isvcReplicas(1)
	suspended.Spec.Suspend = true
	if !memberSuspended(suspended) {
		t.Error("memberSuspended(suspend=true) = false, want true")
	}
	if memberSuspended(isvcReplicas(1)) {
		t.Error("memberSuspended(suspend unset) = true, want false")
	}
}

func TestMemberNamesDeclarationOrder(t *testing.T) {
	pool := mpoolFor("", "", "", "judge", "coder", "gemma")
	got := memberNames(pool)
	want := []string{"judge", "coder", "gemma"}
	if len(got) != len(want) {
		t.Fatalf("memberNames = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("memberNames[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// readyMember builds a Ready, single-replica member InferenceService.
func readyMember() *inferencev1alpha1.InferenceService {
	isvc := &inferencev1alpha1.InferenceService{Spec: inferencev1alpha1.InferenceServiceSpec{Replicas: ptr.To(int32(1))}}
	isvc.Status.Phase = PhaseReady
	return isvc
}

// reclaimDefault is the fixed spec.default used by reclaimPool; the on-demand
// member is "coder".
const reclaimDefault = "glimmer"

// reclaimPool builds a reclaim-policy pool (default = reclaimDefault) with the
// given resident and an optional idle-since instant. members are all Ready.
func reclaimPool(resident string, idleSince *metav1.Time, reclaimAfter time.Duration, names ...string) (*inferencev1alpha1.ModelPool, map[string]*inferencev1alpha1.InferenceService) {
	memberRefs := make([]inferencev1alpha1.ModelPoolMember, 0, len(names))
	m := make(map[string]*inferencev1alpha1.InferenceService, len(names))
	for _, n := range names {
		memberRefs = append(memberRefs, inferencev1alpha1.ModelPoolMember{InferenceServiceRef: corev1.LocalObjectReference{Name: n}})
		m[n] = readyMember()
	}
	pool := &inferencev1alpha1.ModelPool{
		Spec: inferencev1alpha1.ModelPoolSpec{
			Members:      memberRefs,
			Default:      reclaimDefault,
			SwapPolicy:   inferencev1alpha1.ModelPoolSwapPolicyReclaim,
			ReclaimAfter: &metav1.Duration{Duration: reclaimAfter},
		},
		Status: inferencev1alpha1.ModelPoolStatus{ResidentMember: resident, ResidentIdleSince: idleSince},
	}
	return pool, m
}

func TestApplyReclaimOwner(t *testing.T) {
	longAgo := metav1.NewTime(time.Now().Add(-time.Hour))
	justNow := metav1.NewTime(time.Now())

	t.Run("sticky policy never reclaims", func(t *testing.T) {
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder", "glimmer")
		pool.Spec.SwapPolicy = inferencev1alpha1.ModelPoolSwapPolicySticky
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || requeue != 0 || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("sticky: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})

	t.Run("default already resident is a no-op", func(t *testing.T) {
		pool, members := reclaimPool("glimmer", &longAgo, time.Minute, "coder", "glimmer")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "glimmer")
		if owner != "glimmer" || requeue != 0 || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("default-resident: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})

	t.Run("cross-model activation in flight is not preempted", func(t *testing.T) {
		pool, members := reclaimPool("glimmer", &longAgo, time.Minute, "coder", "glimmer")
		// owner != resident: a fresh activation for coder is already redirecting.
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || requeue != 0 || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("activation-in-flight: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})

	t.Run("busy resident resets the idle clock", func(t *testing.T) {
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder", "glimmer")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return false, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("busy: owner=%q idleSince=%v", owner, pool.Status.ResidentIdleSince)
		}
		if requeue != modelPoolReclaimPoll {
			t.Fatalf("busy: requeue=%v want %v", requeue, modelPoolReclaimPoll)
		}
	})

	t.Run("idle probe failure fails closed", func(t *testing.T) {
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder", "glimmer")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) {
			return false, errIdleUnsupported
		}}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || requeue != modelPoolReclaimPoll {
			t.Fatalf("probe-fail: owner=%q requeue=%v", owner, requeue)
		}
		// The clock is held where it was, not reset: an elapsed window must
		// survive a poll the resident did not answer.
		if pool.Status.ResidentIdleSince == nil || !pool.Status.ResidentIdleSince.Time.Equal(longAgo.Time) {
			t.Fatalf("probe-fail: idleSince=%v want it held at %v", pool.Status.ResidentIdleSince, longAgo.Time)
		}
	})

	t.Run("first idle observation starts the clock without reclaiming", func(t *testing.T) {
		pool, members := reclaimPool("coder", nil, 10*time.Minute, "coder", "glimmer")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" {
			t.Fatalf("clock-start: owner=%q want coder", owner)
		}
		if pool.Status.ResidentIdleSince == nil {
			t.Fatal("clock-start: ResidentIdleSince not set")
		}
		if requeue != modelPoolReclaimPoll { // 10m window caps at the poll interval
			t.Fatalf("clock-start: requeue=%v want %v", requeue, modelPoolReclaimPoll)
		}
	})

	t.Run("idle within the window keeps waiting", func(t *testing.T) {
		pool, members := reclaimPool("coder", &justNow, 10*time.Minute, "coder", "glimmer")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || pool.Status.ResidentIdleSince == nil || requeue != modelPoolReclaimPoll {
			t.Fatalf("within-window: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})

	t.Run("idle beyond the window reclaims to default", func(t *testing.T) {
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder", "glimmer")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "glimmer" {
			t.Fatalf("reclaim: owner=%q want glimmer", owner)
		}
		if requeue != 0 {
			t.Fatalf("reclaim: requeue=%v want 0 (swap machinery takes over)", requeue)
		}
	})

	t.Run("short reclaimAfter polls within the window", func(t *testing.T) {
		if got := pollWithin(5 * time.Second); got != 5*time.Second {
			t.Fatalf("pollWithin(5s)=%v want 5s", got)
		}
		if got := pollWithin(time.Hour); got != modelPoolReclaimPoll {
			t.Fatalf("pollWithin(1h)=%v want %v", got, modelPoolReclaimPoll)
		}
	})
}

// TestApplyReclaimOwnerRejectsUnsettledPool covers the branches that reset the
// idle clock and never reclaim because the pool is not settled on a reclaimable
// non-default resident.
func TestApplyReclaimOwnerRejectsUnsettledPool(t *testing.T) {
	longAgo := metav1.NewTime(time.Now().Add(-time.Hour))

	t.Run("empty spec.default resets and never reclaims", func(t *testing.T) {
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder", "glimmer")
		pool.Spec.Default = ""
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || requeue != 0 || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("empty-default: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})

	t.Run("resident not Ready resets and never reclaims", func(t *testing.T) {
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder", "glimmer")
		members["coder"].Status.Phase = "" // anything but PhaseReady is not-Ready
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || requeue != 0 || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("resident-not-ready: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})

	t.Run("default absent from members resets and never reclaims", func(t *testing.T) {
		// spec.default ("glimmer") is not among the resolved members, so there is
		// nothing to reclaim to.
		pool, members := reclaimPool("coder", &longAgo, time.Minute, "coder")
		r := &ModelPoolReconciler{IdleCheck: func(context.Context, *inferencev1alpha1.InferenceService) (bool, error) { return true, nil }}
		owner, requeue := r.applyReclaimOwner(context.Background(), pool, members, "coder")
		if owner != "coder" || requeue != 0 || pool.Status.ResidentIdleSince != nil {
			t.Fatalf("default-not-member: owner=%q requeue=%v idleSince=%v", owner, requeue, pool.Status.ResidentIdleSince)
		}
	})
}
