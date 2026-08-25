package causality

import (
	"math/rand"
	"testing"
)

func TestTransferEntropy(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 200)
	y := make([]float64, 200)
	for i := 1; i < 200; i++ {
		x[i] = 0.5*x[i-1] + rng.NormFloat64()
		y[i] = 0.8*x[i-1] + 0.1*y[i-1] + rng.NormFloat64()
	}
	result, err := TransferEntropy(x, y, 1, 8)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	_ = result.TE
}

func TestConvergentCrossMapping(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float64, 100)
	y := make([]float64, 100)
	for i := 1; i < 100; i++ {
		x[i] = 0.9*x[i-1] + rng.NormFloat64()*0.1
		y[i] = 0.3*x[i-1] + 0.6*y[i-1] + rng.NormFloat64()*0.1
	}
	result, err := ConvergentCrossMapping(x, y, 2, 3)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Rho < -1 || result.Rho > 1 {
		t.Fatalf("correlation out of range: %v", result.Rho)
	}
}

func TestInstantaneousCausality(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 50)
	y := make([]float64, 50)
	for i := range x {
		x[i] = rng.NormFloat64()
		y[i] = x[i] + rng.NormFloat64()*0.1
	}
	result, err := InstantaneousCausality(x, y)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Statistic < 0 {
		t.Fatalf("stat should be non-negative: %v", result.Statistic)
	}
}
