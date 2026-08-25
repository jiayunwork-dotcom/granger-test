package series

import (
	"testing"
)

func TestBuildLagMatrix(t *testing.T) {
	target := []float64{0, 1, 2, 3, 4, 5}
	pred := []float64{10, 11, 12, 13, 14, 15}
	lag := 2

	X, Y, err := BuildLagMatrix(target, pred, lag)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := len(target) - lag
	if len(X) != m || len(Y) != m {
		t.Fatalf("expected %d rows, got X=%d Y=%d", m, len(X), len(Y))
	}

	cols := 1 + 2*lag
	for i, row := range X {
		if len(row) != cols {
			t.Fatalf("row %d: expected %d columns, got %d", i, cols, len(row))
		}
	}

	if Y[0] != target[lag] {
		t.Fatalf("expected Y[0]=%v, got %v", target[lag], Y[0])
	}
}
