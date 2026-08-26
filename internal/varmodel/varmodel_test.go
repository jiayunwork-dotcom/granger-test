package varmodel

import (
	"math/rand"
	"testing"
)

func genVARData(n int, rng *rand.Rand) [][]float64 {
	data := make([][]float64, n)
	data[0] = []float64{0, 0}
	for i := 1; i < n; i++ {
		y0 := 0.5*data[i-1][0] + 0.2*data[i-1][1] + rng.NormFloat64()*0.1
		y1 := 0.3*data[i-1][1] + rng.NormFloat64()*0.1
		data[i] = []float64{y0, y1}
	}
	return data
}

func TestFit_Basic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := genVARData(100, rng)
	model, err := Fit(data, 1)
	if err != nil {
		t.Fatalf("fit failed: %v", err)
	}
	if model.Order != 1 || model.NVars != 2 {
		t.Fatalf("wrong model params")
	}
	if len(model.Coefs) != 2 {
		t.Fatalf("expected 2 coef vectors")
	}
}

func TestForecast(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := genVARData(100, rng)
	model, _ := Fit(data, 2)
	forecasts, err := model.Forecast(data, 5)
	if err != nil {
		t.Fatalf("forecast failed: %v", err)
	}
	if len(forecasts) != 5 {
		t.Fatalf("expected 5 forecasts, got %d", len(forecasts))
	}
}

func TestSelectOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := genVARData(200, rng)
	order, err := SelectOrder(data, 5)
	if err != nil {
		t.Fatalf("select order failed: %v", err)
	}
	if order < 1 || order > 5 {
		t.Fatalf("order out of range: %d", order)
	}
}

func TestGrangerCausality_XtoY(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := genVARData(200, rng)
	fStat, _, err := GrangerCausality(data, 2, 1, 0)
	if err != nil {
		t.Fatalf("granger failed: %v", err)
	}
	if fStat <= 0 {
		t.Fatalf("F stat should be positive: %v", fStat)
	}
}
