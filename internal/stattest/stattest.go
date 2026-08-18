// Package stattest 提供基础统计检验工具。
package stattest

import "math"

// TTest 单样本 t 检验，返回 t 统计量和双侧 p 值。
func TTest(data []float64, mu0 float64) (tStat, pValue float64) {
	n := float64(len(data))
	if n < 2 {
		return 0, 1
	}
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= n
	variance := 0.0
	for _, v := range data {
		d := v - mean
		variance += d * d
	}
	variance /= (n - 1)
	se := math.Sqrt(variance / n)
	if se == 0 {
		return 0, 1
	}
	tStat = (mean - mu0) / se
	df := n - 1
	pValue = 2 * tCDF(-math.Abs(tStat), df)
	return tStat, pValue
}

// TwoSampleT 双样本 t 检验（Welch）。
func TwoSampleT(a, b []float64) (tStat, pValue float64) {
	na := float64(len(a))
	nb := float64(len(b))
	if na < 2 || nb < 2 {
		return 0, 1
	}
	meanA := mean(a)
	meanB := mean(b)
	varA := variance(a, meanA)
	varB := variance(b, meanB)
	se := math.Sqrt(varA/na + varB/nb)
	if se == 0 {
		return 0, 1
	}
	tStat = (meanA - meanB) / se
	v1 := varA / na
	v2 := varB / nb
	df := (v1 + v2) * (v1 + v2) / (v1*v1/(na-1) + v2*v2/(nb-1))
	pValue = 2 * tCDF(-math.Abs(tStat), df)
	return tStat, pValue
}

// FTest F 检验，返回 F 统计量和 p 值。
func FTest(rssRestricted, rssUnrestricted float64, dfRestriction, dfResidual int) (fStat, pValue float64) {
	df1 := float64(dfRestriction)
	df2 := float64(dfResidual)
	if df1 <= 0 || df2 <= 0 || rssUnrestricted <= 0 {
		return 0, 1
	}
	fStat = ((rssRestricted - rssUnrestricted) / df1) / (rssUnrestricted / df2)
	if fStat < 0 {
		fStat = 0
	}
	pValue = fPValue(fStat, df1, df2)
	return fStat, pValue
}

// ChiSquareTest 卡方检验。
func ChiSquareTest(observed, expected []float64) (chiSq, pValue float64) {
	n := len(observed)
	if n == 0 || len(expected) != n {
		return 0, 1
	}
	for i := 0; i < n; i++ {
		if expected[i] > 0 {
			d := observed[i] - expected[i]
			chiSq += d * d / expected[i]
		}
	}
	df := float64(n - 1)
	pValue = 1 - chiSquareCDF(chiSq, df)
	return chiSq, pValue
}

// DurbinWatson 计算 Durbin-Watson 统计量。
func DurbinWatson(residuals []float64) float64 {
	n := len(residuals)
	if n < 2 {
		return 2
	}
	num := 0.0
	den := 0.0
	for i := 1; i < n; i++ {
		d := residuals[i] - residuals[i-1]
		num += d * d
		den += residuals[i] * residuals[i]
	}
	den += residuals[0] * residuals[0]
	if den == 0 {
		return 2
	}
	return num / den
}

// JarqueBera 正态性检验。
func JarqueBera(data []float64) (jbStat, pValue float64) {
	n := float64(len(data))
	if n < 4 {
		return 0, 1
	}
	m := mean(data)
	var m2, m3, m4 float64
	for _, v := range data {
		d := v - m
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	m2 /= n
	m3 /= n
	m4 /= n
	if m2 == 0 {
		return 0, 1
	}
	skew := m3 / math.Pow(m2, 1.5)
	kurt := m4/(m2*m2) - 3
	jbStat = n / 6 * (skew*skew + kurt*kurt/4)
	pValue = 1 - chiSquareCDF(jbStat, 2)
	return jbStat, pValue
}

// LjungBox Ljung-Box 检验。
func LjungBox(acf []float64, n, lags int) (qStat, pValue float64) {
	if lags <= 0 || n <= 0 {
		return 0, 1
	}
	for k := 1; k <= lags && k < len(acf); k++ {
		qStat += acf[k] * acf[k] / float64(n-k)
	}
	qStat *= float64(n * (n + 2))
	pValue = 1 - chiSquareCDF(qStat, float64(lags))
	return qStat, pValue
}

func mean(data []float64) float64 {
	s := 0.0
	for _, v := range data {
		s += v
	}
	return s / float64(len(data))
}

func variance(data []float64, m float64) float64 {
	s := 0.0
	for _, v := range data {
		d := v - m
		s += d * d
	}
	return s / float64(len(data)-1)
}

func tCDF(t, df float64) float64 {
	x := df / (df + t*t)
	return 0.5 * betaInc(x, df/2, 0.5)
}

func fPValue(f, d1, d2 float64) float64 {
	if f <= 0 {
		return 1
	}
	x := d1 * f / (d1*f + d2)
	return 1 - betaInc(x, d1/2, d2/2)
}

func chiSquareCDF(x, df float64) float64 {
	if x <= 0 {
		return 0
	}
	return gammaInc(df/2, x/2)
}

func betaInc(x, a, b float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	la, _ := math.Lgamma(a)
	lb, _ := math.Lgamma(b)
	lab, _ := math.Lgamma(a + b)
	bt := math.Exp(lab - la - lb + a*math.Log(x) + b*math.Log(1-x))
	if x < (a+1)/(a+b+2) {
		return bt * betaCF(a, b, x) / a
	}
	return 1 - bt*betaCF(b, a, 1-x)/b
}

func betaCF(a, b, x float64) float64 {
	const maxit = 200
	const eps = 1e-14
	qab := a + b
	c := 1.0
	d := 1 - qab*x/(a+1)
	if math.Abs(d) < 1e-300 {
		d = 1e-300
	}
	d = 1 / d
	h := d
	for m := 1; m <= maxit; m++ {
		mf := float64(m)
		num := mf * (b - mf) * x / ((a + 2*mf - 1) * (a + 2*mf))
		d = 1 + num*d
		if math.Abs(d) < 1e-300 {
			d = 1e-300
		}
		c = 1 + num/c
		if math.Abs(c) < 1e-300 {
			c = 1e-300
		}
		d = 1 / d
		h *= d * c
		num = -(a + mf) * (qab + mf) * x / ((a + 2*mf) * (a + 2*mf + 1))
		d = 1 + num*d
		if math.Abs(d) < 1e-300 {
			d = 1e-300
		}
		c = 1 + num/c
		if math.Abs(c) < 1e-300 {
			c = 1e-300
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

func gammaInc(a, x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x < a+1 {
		return gammaIncSeries(a, x)
	}
	return 1 - gammaIncCF(a, x)
}

func gammaIncSeries(a, x float64) float64 {
	la, _ := math.Lgamma(a)
	sum := 1.0 / a
	term := 1.0 / a
	for n := 1; n < 200; n++ {
		term *= x / (a + float64(n))
		sum += term
		if math.Abs(term) < 1e-14*math.Abs(sum) {
			break
		}
	}
	return sum * math.Exp(-x+a*math.Log(x)-la)
}

func gammaIncCF(a, x float64) float64 {
	la, _ := math.Lgamma(a)
	b := x + 1 - a
	c := 1.0 / 1e-300
	d := 1 / b
	h := d
	for i := 1; i < 200; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < 1e-300 {
			d = 1e-300
		}
		c = b + an/c
		if math.Abs(c) < 1e-300 {
			c = 1e-300
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < 1e-14 {
			break
		}
	}
	return h * math.Exp(-x+a*math.Log(x)-la)
}
