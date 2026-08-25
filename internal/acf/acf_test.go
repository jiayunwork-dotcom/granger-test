package acf

import (
	"math"
	"testing"
)

func TestACF_Lag0(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	a := ACF(data, 3)
	if a[0] != 1 {
		t.Fatalf("ACF at lag 0 should be 1, got %v", a[0])
	}
}

func TestACF_WhiteNoise(t *testing.T) {
	data := []float64{1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
		1, -1, 1, -1, 1, -1, 1, -1, 1, -1}
	a := ACF(data, 5)
	if a[1] > 0 {
		t.Fatalf("alternating series lag-1 ACF should be negative, got %v", a[1])
	}
}

func TestPACF_AR1(t *testing.T) {
	data := make([]float64, 100)
	data[0] = 1
	for i := 1; i < 100; i++ {
		data[i] = 0.8*data[i-1] + float64(i%3-1)*0.1
	}
	p := PACF(data, 5)
	if len(p) < 3 {
		t.Fatalf("PACF too short")
	}
	if math.Abs(p[1]) < 0.3 {
		t.Fatalf("PACF lag 1 expected high, got %v", p[1])
	}
}

func TestCrossCorrelation(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{1, 2, 3, 4, 5}
	cc := CrossCorrelation(x, y, 2)
	if math.Abs(cc[2]-1.0) > 1e-10 {
		t.Fatalf("perfect correlation at lag 0 expected 1, got %v", cc[2])
	}
}

func TestConfidenceBand(t *testing.T) {
	band := ConfidenceBand(100)
	if math.Abs(band-0.196) > 0.01 {
		t.Fatalf("expected ~0.196, got %v", band)
	}
}

func TestSignificantLags(t *testing.T) {
	acfVals := []float64{1, 0.9, 0.01, 0.8, 0.02}
	lags := SignificantLags(acfVals, 100)
	if len(lags) != 2 || lags[0] != 1 || lags[1] != 3 {
		t.Fatalf("expected lags [1,3], got %v", lags)
	}
}
