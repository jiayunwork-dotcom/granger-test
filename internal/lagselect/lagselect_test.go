package lagselect

import (
	"math/rand"
	"testing"
)

func TestFPE(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	y := make([]float64, 100)
	for i := 1; i < 100; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	results, bestLag, err := FPE(y, 5)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(results) == 0 || bestLag < 1 || bestLag > 5 {
		t.Fatalf("invalid results: lag=%d, n=%d", bestLag, len(results))
	}
}

func TestOptimalLag(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	y := make([]float64, 100)
	for i := 1; i < 100; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	result, err := OptimalLag(y, 5)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.OptimalLag < 1 || result.OptimalLag > 5 {
		t.Fatalf("lag out of range: %d", result.OptimalLag)
	}
}
