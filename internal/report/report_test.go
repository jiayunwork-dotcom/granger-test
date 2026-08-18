package report

import (
	"strings"
	"testing"
)

func TestFormatGranger(t *testing.T) {
	r := []GrangerResult{{
		Cause:      "X",
		Effect:     "Y",
		Lag:        2,
		FStatistic: 5.5,
		PValue:     0.02,
	}}
	s := FormatGranger(r)
	if !strings.Contains(s, "X") {
		t.Fatalf("should contain cause: %s", s)
	}
}

func TestFormatADF(t *testing.T) {
	r := []ADFResult{{
		Variable:   "y",
		Statistic:  -3.5,
		PValue:     0.01,
		Lags:       3,
		Stationary: true,
	}}
	s := FormatADF(r)
	if !strings.Contains(s, "y") {
		t.Fatalf("should contain variable name: %s", s)
	}
}

func TestFormatJSON(t *testing.T) {
	data := map[string]float64{"f": 5.5, "p": 0.02}
	s, err := FormatJSON(data)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !strings.Contains(s, "5.5") {
		t.Fatalf("JSON should contain value: %s", s)
	}
}
