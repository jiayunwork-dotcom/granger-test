package granger

import (
	"math"
	"testing"
)

func TestGrangerCausal(t *testing.T) {
	// y is driven by x with lag 2: y[t] = x[t-2] plus tiny noise
	n := 120
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) * 0.3)
	}
	for i := 2; i < n; i++ {
		y[i] = x[i-2] + 0.001*math.Sin(float64(i))
	}

	res, err := Test(x, y, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.FX <= res.FY {
		t.Fatalf("expected X->Y F (%.3f) larger than Y->X F (%.3f)", res.FX, res.FY)
	}
	if res.PX >= 0.05 {
		t.Fatalf("expected a small p-value for the causal direction, got %.4f", res.PX)
	}
	if res.Direction != "X→Y" {
		t.Fatalf("expected direction X→Y, got %s", res.Direction)
	}
}

func TestGrangerIndependent(t *testing.T) {
	// two unrelated deterministic series
	n := 400
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) * 0.3)
		y[i] = math.Cos(float64(i) * 0.7)
	}

	res, err := Test(x, y, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.PX < 0.05 {
		t.Fatalf("expected a large p-value for X->Y on independent data, got %.4f", res.PX)
	}
	if res.PY < 0.05 {
		t.Fatalf("expected a large p-value for Y->X on independent data, got %.4f", res.PY)
	}
}

func TestGrangerInvalidLag(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{5, 4, 3, 2, 1}

	if _, err := Test(x, y, 0); err == nil {
		t.Fatal("expected error for lag=0, got nil")
	}
	if _, err := Test(x, y, -1); err == nil {
		t.Fatal("expected error for negative lag, got nil")
	}
	if _, err := Test(x, y, 5); err == nil {
		t.Fatal("expected error for lag >= series length, got nil")
	}
}
