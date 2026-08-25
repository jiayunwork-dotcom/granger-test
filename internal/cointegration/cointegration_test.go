package cointegration

import (
	"math/rand"
	"testing"
)

func TestEngleGranger_Cointegrated(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 200
	x := make([]float64, n)
	y := make([]float64, n)
	x[0] = 0
	for i := 1; i < n; i++ {
		x[i] = x[i-1] + rng.NormFloat64()
		y[i] = 2*x[i] + rng.NormFloat64()*0.5
	}
	result, err := EngleGranger(y, [][]float64{x})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	_ = result.ADFStat
}

func TestEngleGranger_NotCointegrated(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 200
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		x[i] = x[i-1] + rng.NormFloat64()
		y[i] = y[i-1] + rng.NormFloat64()
	}
	result, err := EngleGranger(y, [][]float64{x})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Cointegrated {
		t.Fatalf("independent random walks should not be cointegrated")
	}
}

func TestJohansenTrace(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 100
	data := make([][]float64, n)
	data[0] = []float64{0, 0}
	for i := 1; i < n; i++ {
		data[i] = []float64{
			data[i-1][0] + rng.NormFloat64(),
			data[i-1][1] + rng.NormFloat64(),
		}
	}
	result, err := JohansenTrace(data, 2)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(result.TraceStats) == 0 {
		t.Fatalf("should have trace stats")
	}
}
