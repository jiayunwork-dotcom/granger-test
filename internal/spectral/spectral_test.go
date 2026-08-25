package spectral

import (
	"math"
	"testing"
)

func TestDFT_DC(t *testing.T) {
	x := []float64{1, 1, 1, 1}
	dft := DFT(x)
	if math.Abs(dft[0].Re-4) > 1e-10 {
		t.Fatalf("DC component should be 4, got %v", dft[0].Re)
	}
}

func TestPowerSpectrum_Sine(t *testing.T) {
	n := 64
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2 * math.Pi * 4 * float64(i) / float64(n))
	}
	psd := PowerSpectrum(x)
	peak := DominantFreq(psd)
	if peak != 4 {
		t.Fatalf("dominant freq should be 4, got %d", peak)
	}
}

func TestPeriodogram(t *testing.T) {
	x := make([]float64, 32)
	for i := range x {
		x[i] = math.Cos(2*math.Pi*2*float64(i)/32) + 0.5
	}
	p := Periodogram(x)
	if len(p) == 0 {
		t.Fatalf("periodogram should not be empty")
	}
}

func TestCoherence_Same(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	coh := Coherence(x, x)
	for k := 0; k < len(coh); k++ {
		if coh[k] < 0.99 {
			t.Fatalf("self-coherence should be ~1 at freq %d, got %v", k, coh[k])
		}
	}
}

func TestHann(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1}
	h := Hann(x)
	if h[0] != 0 || h[len(h)-1] != 0 {
		t.Fatalf("Hann window edges should be 0")
	}
	if h[2] < 0.9 {
		t.Fatalf("Hann window center should be ~1, got %v", h[2])
	}
}

func TestSpectralDensity(t *testing.T) {
	n := 64
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2*math.Pi*8*float64(i)/float64(n)) + float64(i%3)*0.1
	}
	psd := SpectralDensity(x, 16)
	if len(psd) == 0 {
		t.Fatalf("spectral density should not be empty")
	}
}
