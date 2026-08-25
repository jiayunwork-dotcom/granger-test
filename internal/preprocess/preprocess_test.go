package preprocess

import (
	"math"
	"testing"
)

func TestDiff_Order1(t *testing.T) {
	data := []float64{1, 3, 6, 10}
	d := Diff(data, 1)
	if len(d) != 3 || d[0] != 2 || d[1] != 3 || d[2] != 4 {
		t.Fatalf("diff wrong: %v", d)
	}
}

func TestDiff_Order2(t *testing.T) {
	data := []float64{1, 3, 6, 10, 15}
	d := Diff(data, 2)
	if len(d) != 3 || d[0] != 1 || d[1] != 1 || d[2] != 1 {
		t.Fatalf("2nd diff wrong: %v", d)
	}
}

func TestDetrend(t *testing.T) {
	data := []float64{2, 4, 6, 8, 10}
	dt := Detrend(data)
	for _, v := range dt {
		if math.Abs(v) > 1e-10 {
			t.Fatalf("detrended linear should be 0, got %v", v)
		}
	}
}

func TestStandardize(t *testing.T) {
	data := []float64{10, 20, 30, 40, 50}
	s := Standardize(data)
	mean := 0.0
	for _, v := range s {
		mean += v
	}
	mean /= float64(len(s))
	if math.Abs(mean) > 1e-10 {
		t.Fatalf("standardized mean should be 0, got %v", mean)
	}
}

func TestLogTransform(t *testing.T) {
	data := []float64{1, math.E, math.E * math.E}
	lt := LogTransform(data)
	if math.Abs(lt[0]) > 1e-10 || math.Abs(lt[1]-1) > 1e-10 || math.Abs(lt[2]-2) > 1e-10 {
		t.Fatalf("log wrong: %v", lt)
	}
}

func TestFillNA(t *testing.T) {
	data := []float64{1, math.NaN(), math.NaN(), 4}
	filled := FillNA(data)
	if filled[1] != 1 || filled[2] != 1 {
		t.Fatalf("fill wrong: %v", filled)
	}
}

func TestWinsorize(t *testing.T) {
	data := []float64{-100, 1, 2, 3, 4, 5, 100}
	w := Winsorize(data, 15)
	if w[0] < -100 || w[6] > 100 {
	}
	if w[2] != 2 || w[3] != 3 {
		t.Fatalf("middle values should be unchanged: %v", w)
	}
}

func TestSeasonalDiff(t *testing.T) {
	data := []float64{10, 20, 30, 12, 22, 32}
	sd := SeasonalDiff(data, 3)
	if len(sd) != 3 || sd[0] != 2 || sd[1] != 2 {
		t.Fatalf("seasonal diff wrong: %v", sd)
	}
}

func TestBoxCox_Lambda0(t *testing.T) {
	data := []float64{1, math.E, math.E * math.E}
	bc := BoxCox(data, 0)
	if math.Abs(bc[1]-1) > 1e-10 {
		t.Fatalf("BoxCox(e,0) should be 1, got %v", bc[1])
	}
}

func TestMovingMedian(t *testing.T) {
	data := []float64{1, 100, 3, 4, 5}
	mm := MovingMedian(data, 3)
	if mm[1] != 3 {
		t.Fatalf("moving median at index 1 should be 3, got %v", mm[1])
	}
}
