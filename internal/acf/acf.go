package acf

import "math"

func ACF(data []float64, maxLag int) []float64 {
	n := len(data)
	if n == 0 || maxLag <= 0 {
		return nil
	}
	if maxLag >= n {
		maxLag = n - 1
	}
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(n)
	var c0 float64
	for _, v := range data {
		d := v - mean
		c0 += d * d
	}
	if c0 == 0 {
		return make([]float64, maxLag+1)
	}
	result := make([]float64, maxLag+1)
	result[0] = 1
	for lag := 1; lag <= maxLag; lag++ {
		cov := 0.0
		for i := lag; i < n; i++ {
			cov += (data[i] - mean) * (data[i-lag] - mean)
		}
		result[lag] = cov / c0
	}
	return result
}

func PACF(data []float64, maxLag int) []float64 {
	acfVals := ACF(data, maxLag)
	if len(acfVals) <= 1 {
		return nil
	}
	n := len(acfVals) - 1
	pacf := make([]float64, n+1)
	pacf[0] = 1
	if n == 0 {
		return pacf
	}
	phi := make([][]float64, n+1)
	for i := range phi {
		phi[i] = make([]float64, n+1)
	}
	phi[1][1] = acfVals[1]
	pacf[1] = acfVals[1]
	for k := 2; k <= n; k++ {
		num := acfVals[k]
		for j := 1; j < k; j++ {
			num -= phi[k-1][j] * acfVals[k-j]
		}
		denom := 1.0
		for j := 1; j < k; j++ {
			denom -= phi[k-1][j] * acfVals[j]
		}
		if denom == 0 {
			break
		}
		phi[k][k] = num / denom
		pacf[k] = phi[k][k]
		for j := 1; j < k; j++ {
			phi[k][j] = phi[k-1][j] - phi[k][k]*phi[k-1][k-j]
		}
	}
	return pacf
}

func CrossCorrelation(x, y []float64, maxLag int) []float64 {
	n := len(x)
	if n == 0 || len(y) != n || maxLag <= 0 {
		return nil
	}
	if maxLag >= n {
		maxLag = n - 1
	}
	meanX := 0.0
	meanY := 0.0
	for i := 0; i < n; i++ {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(n)
	meanY /= float64(n)
	varX := 0.0
	varY := 0.0
	for i := 0; i < n; i++ {
		varX += (x[i] - meanX) * (x[i] - meanX)
		varY += (y[i] - meanY) * (y[i] - meanY)
	}
	denom := math.Sqrt(varX * varY)
	if denom == 0 {
		return make([]float64, 2*maxLag+1)
	}
	result := make([]float64, 2*maxLag+1)
	for lag := -maxLag; lag <= maxLag; lag++ {
		cov := 0.0
		count := 0
		for i := 0; i < n; i++ {
			j := i + lag
			if j >= 0 && j < n {
				cov += (x[i] - meanX) * (y[j] - meanY)
				count++
			}
		}
		result[lag+maxLag] = cov / denom
	}
	return result
}

func ConfidenceBand(n int) float64 {
	if n <= 0 {
		return 0
	}
	return 1.96 / math.Sqrt(float64(n))
}

func SignificantLags(acfVals []float64, n int) []int {
	band := ConfidenceBand(n)
	var lags []int
	for i := 1; i < len(acfVals); i++ {
		if math.Abs(acfVals[i]) > band {
			lags = append(lags, i)
		}
	}
	return lags
}
