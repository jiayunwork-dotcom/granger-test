package adf

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

type Result struct {
	ADFStat    float64
	PValue     float64
	Lags       int
	Stationary bool
}

func Test(series []float64, maxLag int) (Result, error) {
	n := len(series)
	if n < 10 {
		return Result{}, fmt.Errorf("series too short: %d", n)
	}
	if maxLag <= 0 {
		maxLag = int(math.Floor(math.Pow(float64(n-1), 1.0/3.0)))
	}
	if maxLag >= n/2 {
		maxLag = n/2 - 1
	}

	dy := make([]float64, n-1)
	for i := 1; i < n; i++ {
		dy[i-1] = series[i] - series[i-1]
	}

	bestAIC := math.Inf(1)
	var bestResult Result

	for lag := 0; lag <= maxLag; lag++ {
		stat, aic, err := runADF(series, dy, lag)
		if err != nil {
			continue
		}
		if aic < bestAIC {
			bestAIC = aic
			bestResult = Result{
				ADFStat:    stat,
				Lags:       lag,
				PValue:     adfPValue(stat, n),
				Stationary: stat < criticalValue5(n),
			}
		}
	}
	return bestResult, nil
}

func runADF(y, dy []float64, lag int) (float64, float64, error) {
	n := len(dy)
	start := lag + 1
	if start >= n {
		return 0, 0, fmt.Errorf("not enough data")
	}
	m := n - start
	k := 2 + lag
	X := make([][]float64, m)
	Y := make([]float64, m)
	for i := 0; i < m; i++ {
		t := start + i
		row := make([]float64, k)
		row[0] = 1
		row[1] = y[t]
		for l := 0; l < lag; l++ {
			if t-1-l >= 0 && t-1-l < len(dy) {
				row[2+l] = dy[t-1-l]
			}
		}
		X[i] = row
		Y[i] = dy[t]
	}
	beta, rss, err := ols.Fit(X, Y)
	if err != nil {
		return 0, 0, err
	}
	sigma2 := rss / float64(m-k)
	if sigma2 <= 0 {
		return 0, 0, fmt.Errorf("zero variance")
	}
	xtx11 := 0.0
	for i := 0; i < m; i++ {
		xtx11 += X[i][1] * X[i][1]
	}
	if xtx11 == 0 {
		return 0, 0, fmt.Errorf("zero variance in regressor")
	}
	se := math.Sqrt(sigma2 / xtx11)
	tStat := beta[1] / se
	aic := float64(m)*math.Log(rss/float64(m)) + 2*float64(k)
	return tStat, aic, nil
}

func adfPValue(stat float64, n int) float64 {
	cv1 := -3.43
	cv5 := -2.86
	cv10 := -2.57
	if stat < cv1 {
		return 0.01
	}
	if stat < cv5 {
		return 0.05
	}
	if stat < cv10 {
		return 0.10
	}
	if stat < 0 {
		return 0.1 + 0.9*(-stat)/(-cv10)
	}
	return 0.99
}

func criticalValue5(_ int) float64 {
	return -2.86
}

func KPSS(series []float64) (float64, error) {
	n := len(series)
	if n < 5 {
		return 0, fmt.Errorf("series too short")
	}
	mean := 0.0
	for _, v := range series {
		mean += v
	}
	mean /= float64(n)
	residuals := make([]float64, n)
	for i, v := range series {
		residuals[i] = v - mean
	}
	cumSum := make([]float64, n)
	cumSum[0] = residuals[0]
	for i := 1; i < n; i++ {
		cumSum[i] = cumSum[i-1] + residuals[i]
	}
	bw := int(math.Floor(4 * math.Pow(float64(n)/100, 2.0/9.0)))
	sigma2 := 0.0
	for i := 0; i < n; i++ {
		sigma2 += residuals[i] * residuals[i]
	}
	sigma2 /= float64(n)
	for l := 1; l <= bw; l++ {
		w := 1 - float64(l)/float64(bw+1)
		gamma := 0.0
		for i := l; i < n; i++ {
			gamma += residuals[i] * residuals[i-l]
		}
		gamma /= float64(n)
		sigma2 += 2 * w * gamma
	}
	if sigma2 <= 0 {
		return 0, fmt.Errorf("non-positive variance estimate")
	}
	stat := 0.0
	for i := 0; i < n; i++ {
		stat += cumSum[i] * cumSum[i]
	}
	stat /= float64(n) * float64(n) * sigma2
	return stat, nil
}
