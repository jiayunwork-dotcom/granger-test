package window

import (
	"math"
	"testing"
)

func TestHammingWindow(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1}
	h := HammingWindow(x)
	if len(h) != 5 {
		t.Fatalf("wrong length")
	}
	if h[2] < 0.9 {
		t.Fatalf("center should be near 1, got %v", h[2])
	}
}

func TestBlackmanWindow(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1}
	b := BlackmanWindow(x)
	if math.Abs(b[0]) > 0.01 {
		t.Fatalf("edge should be ~0, got %v", b[0])
	}
}

func TestRollingWindow(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	wins := RollingWindow(data, 3, 1)
	if len(wins) != 4 {
		t.Fatalf("expected 4 windows, got %d", len(wins))
	}
}

func TestExpandingWindow(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	wins := ExpandingWindow(data, 2)
	if len(wins) != 4 {
		t.Fatalf("expected 4 windows, got %d", len(wins))
	}
	if len(wins[3]) != 5 {
		t.Fatalf("last window should be full size")
	}
}
