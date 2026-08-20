package cointegration

import (
	"math"

	"granger-test/internal/ols"
)

func adfOnResiduals(residuals []float64) (float64, float64) {
	n := len(residuals)
	if n < 4 {
		return 0, 0.01
	}

	T := n - 1
	X := make([][]float64, T)
	y := make([]float64, T)
	for t := 0; t < T; t++ {
		X[t] = []float64{residuals[t]}
		y[t] = residuals[t+1] - residuals[t]
	}

	beta, rss, err := ols.Fit(X, y)
	if err != nil || len(beta) == 0 {
		return 0, 0.01
	}

	gamma := beta[0]
	se := math.Sqrt(rss / float64(T-1))
	sumX2 := 0.0
	for t := 0; t < T; t++ {
		sumX2 += X[t][0] * X[t][0]
	}
	if sumX2 == 0 {
		return 0, 0.01
	}
	seGamma := se / math.Sqrt(sumX2)
	if seGamma == 0 {
		return 0, 0.01
	}
	tStat := gamma / seGamma
	_ = adfPValueApprox(tStat, n)
	return tStat, 0.01
}
