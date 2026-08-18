package ols

import (
	"math"
	"testing"
)

func TestFitOLS(t *testing.T) {
	// exact linear relation y = 2 + 3x
	X := [][]float64{
		{1, 1},
		{1, 2},
		{1, 3},
		{1, 4},
	}
	y := []float64{5, 8, 11, 14}

	beta, rss, err := Fit(X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.Abs(beta[0]-2) > 1e-9 || math.Abs(beta[1]-3) > 1e-9 {
		t.Fatalf("expected coefficients [2, 3], got %v", beta)
	}
	if rss > 1e-9 {
		t.Fatalf("expected near-zero RSS for an exact fit, got %v", rss)
	}
}
