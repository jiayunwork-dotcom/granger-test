package stattest

import (
	"math"
	"testing"
)

func TestTTest_ZeroMean(t *testing.T) {
	data := []float64{-1, 1, -2, 2, 0, 0}
	tStat, p := TTest(data, 0)
	if math.Abs(tStat) > 0.01 {
		t.Fatalf("expected t near 0, got %v", tStat)
	}
	if p < 0.9 {
		t.Fatalf("expected high p-value, got %v", p)
	}
}

func TestTwoSampleT(t *testing.T) {
	a := []float64{10, 11, 12, 13, 14}
	b := []float64{20, 21, 22, 23, 24}
	_, p := TwoSampleT(a, b)
	if p > 0.001 {
		t.Fatalf("clearly different samples should have small p, got %v", p)
	}
}

func TestFTest(t *testing.T) {
	f, p := FTest(100, 80, 2, 50)
	if f <= 0 {
		t.Fatalf("F should be positive")
	}
	if p <= 0 || p >= 1 {
		t.Fatalf("p-value out of range: %v", p)
	}
}

func TestDurbinWatson(t *testing.T) {
	resid := []float64{1, -1, 1, -1, 1, -1, 1, -1}
	dw := DurbinWatson(resid)
	if dw < 1.5 || dw > 4.5 {
		t.Fatalf("alternating residuals DW expected 1.5-4.5, got %v", dw)
	}
}

func TestJarqueBera_Normal(t *testing.T) {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i%10 - 5)
	}
	_, p := JarqueBera(data)
	if p > 0.9 {
		t.Fatalf("uniform-like data JB p expected low, got %v", p)
	}
}

func TestChiSquareTest(t *testing.T) {
	obs := []float64{50, 30, 20}
	exp := []float64{33.3, 33.3, 33.4}
	chi, p := ChiSquareTest(obs, exp)
	if chi <= 0 {
		t.Fatalf("chi should be positive")
	}
	if p >= 1 || p <= 0 {
		t.Fatalf("p out of range: %v", p)
	}
}

func TestLjungBox(t *testing.T) {
	acf := []float64{1, 0.5, 0.3, 0.1}
	q, p := LjungBox(acf, 100, 3)
	if q <= 0 {
		t.Fatalf("Q should be positive")
	}
	if p <= 0 || p >= 1 {
		t.Fatalf("p out of range: %v", p)
	}
}
