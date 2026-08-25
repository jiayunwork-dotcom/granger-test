package residual

import (
	"math/rand"
	"testing"
)

func TestBreuschGodfrey(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	resid := make([]float64, 100)
	for i := range resid {
		resid[i] = rng.NormFloat64()
	}
	X := make([][]float64, 100)
	for i := range X {
		X[i] = []float64{1, float64(i)}
	}
	result, err := BreuschGodfrey(resid, X, 3)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Statistic < 0 || result.PValue < 0 || result.PValue > 1 {
		t.Fatalf("invalid BG result: stat=%v, p=%v", result.Statistic, result.PValue)
	}
}

func TestArchTest(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	resid := make([]float64, 100)
	for i := range resid {
		resid[i] = rng.NormFloat64()
	}
	result, err := ArchTest(resid, 5)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Statistic < 0 || result.PValue < 0 || result.PValue > 1 {
		t.Fatalf("invalid ArchTest result: stat=%v, p=%v", result.Statistic, result.PValue)
	}
}

func TestNormalityTest(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	resid := make([]float64, 100)
	for i := range resid {
		resid[i] = rng.NormFloat64()
	}
	result, err := NormalityTest(resid)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Statistic < 0 {
		t.Fatalf("JB stat should be non-negative: %v", result.Statistic)
	}
}
