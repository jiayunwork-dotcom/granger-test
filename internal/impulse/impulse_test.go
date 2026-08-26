package impulse

import "testing"

func TestOrthoIRF(t *testing.T) {
	coefs := [][][]float64{
		{{0.5, 0.2}, {0.0, 0.3}},
	}
	sigma := [][]float64{{1, 0}, {0, 1}}
	result, err := OrthoIRF(coefs, sigma, 10)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Horizon != 10 {
		t.Fatalf("expected horizon=10, got %d", result.Horizon)
	}
}

func TestFEVD(t *testing.T) {
	coefs := [][][]float64{
		{{0.5, 0.2}, {0.0, 0.3}},
	}
	sigma := [][]float64{{1, 0}, {0, 1}}
	result, err := FEVD(coefs, sigma, 10)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Horizon != 10 {
		t.Fatalf("expected horizon=10, got %d", result.Horizon)
	}
}

func TestCumulativeIRF(t *testing.T) {
	coefs := [][][]float64{
		{{0.5, 0.2}, {0.0, 0.3}},
	}
	sigma := [][]float64{{1, 0}, {0, 1}}
	result, err := CumulativeIRF(coefs, sigma, 5)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Horizon != 5 {
		t.Fatalf("expected horizon=5, got %d", result.Horizon)
	}
}
