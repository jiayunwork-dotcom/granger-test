package adf

import (
	"math/rand"
	"testing"
)

func TestADF_Stationary(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 200)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	result, err := Test(data, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Stationary {
		t.Fatalf("white noise should be stationary, stat=%v", result.ADFStat)
	}
}

func TestADF_RandomWalk(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := make([]float64, 200)
	data[0] = 0
	for i := 1; i < len(data); i++ {
		data[i] = data[i-1] + rng.NormFloat64()
	}
	result, err := Test(data, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Stationary {
		t.Fatalf("random walk should be non-stationary, stat=%v", result.ADFStat)
	}
}

func TestKPSS_Stationary(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 100)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	stat, err := KPSS(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat > 1.0 {
		t.Fatalf("stationary data should have small KPSS, got %v", stat)
	}
}
