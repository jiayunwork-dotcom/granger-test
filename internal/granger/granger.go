package granger

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
	"granger-test/internal/series"
)

// Result holds the Granger causality statistics for both directions.
type Result struct {
	FX, FY    float64 // F statistics for X->Y and Y->X
	PX, PY    float64 // upper-tail p-values
	Direction string  // "X→Y", "Y→X", or "无"
}

// Test performs a Granger causality test of lag order `lag` between two
// equal-length series. It returns an error when lag is non-positive or not
// smaller than the series length.
func Test(x, y []float64, lag int) (Result, error) {
	if lag <= 0 {
		return Result{}, fmt.Errorf("lag must be positive, got %d", lag)
	}
	if len(x) != len(y) {
		return Result{}, fmt.Errorf("series length mismatch")
	}
	if lag >= len(x) {
		return Result{}, fmt.Errorf("lag %d must be smaller than series length %d", lag, len(x))
	}

	fx, px := direction(y, x, lag) // does x cause y?
	fy, py := direction(x, y, lag) // does y cause x?
	return Result{
		FX:       fx,
		FY:       fy,
		PX:       px,
		PY:       py,
		Direction: decide(fx, px, fy, py),
	}, nil
}

// direction returns the F statistic and p-value for the hypothesis that
// `predictor` Granger-causes `target`.
func direction(target, predictor []float64, lag int) (float64, float64) {
	Xu, Yu, err := series.BuildLagMatrix(target, predictor, lag)
	if err != nil {
		return 0, 1
	}
	m := len(Yu)

	Xr := make([][]float64, m)
	for i := 0; i < m; i++ {
		row := make([]float64, 1+lag)
		copy(row, Xu[i][:1+lag])
		Xr[i] = row
	}

	_, rssR, errR := ols.Fit(Xr, Yu)
	_, rssU, errU := ols.Fit(Xu, Yu)
	if errR != nil || errU != nil {
		return 0, 1
	}

	df1 := float64(lag)
	df2 := float64(m - (2*lag + 1))
	if df2 <= 0 || rssU <= 0 {
		return 0, 1
	}
	f := ((rssR - rssU) / df1) / (rssU / df2)
	if f < 0 {
		f = 0
	}
	return f, fPValue(f, df1, df2)
}

// decide picks the causal direction using a 0.05 significance level.
func decide(fx, px, fy, py float64) string {
	const alpha = 0.05
	sx := px < alpha
	sy := py < alpha
	switch {
	case sx && !sy:
		return "X→Y"
	case sy && !sx:
		return "Y→X"
	case sx && sy:
		if px <= py {
			return "X→Y"
		}
		return "Y→X"
	default:
		return "无"
	}
}

// fPValue returns the upper-tail p-value of an F statistic with df (d1, d2).
func fPValue(f, d1, d2 float64) float64 {
	if f <= 0 {
		return 1
	}
	x := d1 * f / (d1*f + d2)
	return 1 - betainc(x, d1/2, d2/2)
}

func logBeta(a, b float64) float64 {
	la, _ := math.Lgamma(a)
	lb, _ := math.Lgamma(b)
	lab, _ := math.Lgamma(a + b)
	return la + lb - lab
}

// betainc evaluates the regularized incomplete beta function I_x(a, b) via
// the continued fraction representation.
func betainc(x, a, b float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	bt := math.Exp(math.Log(x)*a+math.Log(1-x)*b-logBeta(a, b)) / a
	if x < (a+1)/(a+b+2) {
		return bt * betacf(a, b, x) / a
	}
	return 1 - bt*betacf(b, a, 1-x)/b
}

func betacf(a, b, x float64) float64 {
	const maxit = 300
	const eps = 1e-14
	const fpmin = 1e-300
	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1.0 - qab*x/qap
	if math.Abs(d) < fpmin {
		d = fpmin
	}
	d = 1 / d
	h := d
	for m := 1; m <= maxit; m++ {
		mf := float64(m)
		m2 := 2 * mf
		aa := mf * (b - mf) * x / ((qam + m2) * (a + m2))
		d = 1 + aa*d
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = 1 + aa/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1 / d
		h *= d * c
		aa = -(a+mf)*(qab+mf)*x/((a+m2)*(qap+m2))
		d = 1 + aa*d
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = 1 + aa/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < eps {
			break
		}
	}
	return h
}
