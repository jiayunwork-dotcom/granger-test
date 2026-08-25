package information

import (
	"math"
	"testing"
)

func TestAIC(t *testing.T) {
	aic := AIC(100, 50, 3)
	if math.IsInf(aic, 0) || math.IsNaN(aic) {
		t.Fatalf("AIC should be finite")
	}
}

func TestBIC_MorePenalty(t *testing.T) {
	aic := AIC(100, 50, 5)
	bic := BIC(100, 50, 5)
	if bic <= aic {
		t.Fatalf("BIC should penalize more than AIC for n=50, k=5")
	}
}

func TestHQIC(t *testing.T) {
	h := HQIC(100, 50, 3)
	if math.IsInf(h, 0) {
		t.Fatalf("HQIC should be finite")
	}
}

func TestAICc_SmallSample(t *testing.T) {
	aicc := AICc(100, 10, 3)
	aic := AIC(100, 10, 3)
	if aicc <= aic {
		t.Fatalf("AICc should be >= AIC for small sample")
	}
}

func TestSelectLag(t *testing.T) {
	rss := []float64{100, 80, 78, 77.5, 77.4}
	lag := SelectLag(rss, 100, "aic")
	if lag < 1 || lag > 5 {
		t.Fatalf("lag out of range: %d", lag)
	}
}

func TestRSquared(t *testing.T) {
	r2 := RSquared(10, 100)
	if math.Abs(r2-0.9) > 1e-10 {
		t.Fatalf("expected 0.9, got %v", r2)
	}
}

func TestAdjRSquared(t *testing.T) {
	adj := AdjRSquared(10, 100, 50, 3)
	r2 := RSquared(10, 100)
	if adj >= r2 {
		t.Fatalf("adj R² should be <= R²")
	}
}

func TestLikelihoodRatio(t *testing.T) {
	lr, p := LikelihoodRatio(-50, -40, 2)
	if lr != 20 {
		t.Fatalf("expected 20, got %v", lr)
	}
	if p <= 0 || p >= 1 {
		t.Fatalf("p out of range: %v", p)
	}
}
