package bootstrap

import (
	"math"
	"math/rand"
	"testing"
)

func TestResidualBootstrap(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	residuals := make([]float64, 100)
	for i := range residuals {
		residuals[i] = rng.NormFloat64()
	}
	fitted := make([]float64, 100)
	for i := range fitted {
		fitted[i] = float64(i) * 0.1
	}
	sample := ResidualBootstrap(fitted, residuals, rng)
	if len(sample) != 100 {
		t.Fatalf("expected 100, got %d", len(sample))
	}
}

func TestBlockBootstrap(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := make([]float64, 100)
	for i := range data {
		data[i] = float64(i)
	}
	sample := BlockBootstrap(data, 10, rng)
	if len(sample) != 100 {
		t.Fatalf("expected 100, got %d", len(sample))
	}
}

func TestConfidenceInterval(t *testing.T) {
	stats := make([]float64, 1000)
	for i := range stats {
		stats[i] = float64(i)
	}
	lo, hi := ConfidenceInterval(stats, 0.95)
	if lo >= hi {
		t.Fatalf("lo should be < hi: %v, %v", lo, hi)
	}
	if lo < 20 || hi > 980 {
		t.Fatalf("CI out of expected range: %v, %v", lo, hi)
	}
}

func TestBootstrapMean(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 50)
	for i := range data {
		data[i] = 10 + rng.NormFloat64()
	}
	lo, hi := BootstrapMean(data, 500, 0.95, rng)
	if lo > 10 || hi < 10 {
		t.Fatalf("CI should contain true mean ~10: [%v, %v]", lo, hi)
	}
}

func TestBootstrapVariance(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := make([]float64, 100)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	se := BootstrapVariance(data, 200, rng)
	if se <= 0 || se > 0.5 {
		t.Fatalf("SE out of range: %v", se)
	}
}

func TestPermutationTest(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 30)
	y := make([]float64, 30)
	for i := range x {
		x[i] = 10 + rng.NormFloat64()
		y[i] = 12 + rng.NormFloat64()
	}
	p := PermutationTest(x, y, 500, rng)
	if p > 0.1 {
		t.Fatalf("clearly different groups should have small p, got %v", p)
	}
}

func TestStationaryBootstrap(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := make([]float64, 50)
	for i := range data {
		data[i] = float64(i)
	}
	sample := StationaryBootstrap(data, 0.1, rng)
	if len(sample) != 50 {
		t.Fatalf("expected 50, got %d", len(sample))
	}
	_ = math.Abs(sample[0])
}
