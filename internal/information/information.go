// Package information 提供信息准则和模型选择工具。
package information

import "math"

// AIC Akaike 信息准则。
func AIC(rss float64, n, k int) float64 {
	nf := float64(n)
	kf := float64(k)
	if nf <= 0 || rss <= 0 {
		return math.Inf(1)
	}
	return nf*math.Log(rss/nf) + 2*kf
}

// BIC 贝叶斯信息准则。
func BIC(rss float64, n, k int) float64 {
	nf := float64(n)
	kf := float64(k)
	if nf <= 0 || rss <= 0 {
		return math.Inf(1)
	}
	return nf*math.Log(rss/nf) + kf*math.Log(nf)
}

// HQIC Hannan-Quinn 信息准则。
func HQIC(rss float64, n, k int) float64 {
	nf := float64(n)
	kf := float64(k)
	if nf <= 0 || rss <= 0 {
		return math.Inf(1)
	}
	return nf*math.Log(rss/nf) + 2*kf*math.Log(math.Log(nf))
}

// AICc 修正 AIC（小样本）。
func AICc(rss float64, n, k int) float64 {
	nf := float64(n)
	kf := float64(k)
	if nf <= kf+1 {
		return math.Inf(1)
	}
	aic := AIC(rss, n, k)
	return aic + 2*kf*(kf+1)/(nf-kf-1)
}

// SelectLag 使用信息准则选择最佳滞后阶数。
func SelectLag(rssValues []float64, n int, criterion string) int {
	best := 0
	bestVal := math.Inf(1)
	for k, rss := range rssValues {
		var val float64
		switch criterion {
		case "bic":
			val = BIC(rss, n, k+1)
		case "hqic":
			val = HQIC(rss, n, k+1)
		default:
			val = AIC(rss, n, k+1)
		}
		if val < bestVal {
			bestVal = val
			best = k
		}
	}
	return best + 1
}

// LikelihoodRatio 似然比检验。
func LikelihoodRatio(logLRestricted, logLUnrestricted float64, dfDiff int) (lrStat, pValue float64) {
	lrStat = 2 * (logLUnrestricted - logLRestricted)
	if lrStat < 0 {
		lrStat = 0
	}
	pValue = 1 - chiSquareCDF(lrStat, float64(dfDiff))
	return lrStat, pValue
}

// RSquared 决定系数。
func RSquared(rss, tss float64) float64 {
	if tss <= 0 {
		return 0
	}
	return 1 - rss/tss
}

// AdjRSquared 调整的 R²。
func AdjRSquared(rss, tss float64, n, k int) float64 {
	r2 := RSquared(rss, tss)
	nf := float64(n)
	kf := float64(k)
	if nf-kf-1 <= 0 {
		return 0
	}
	return 1 - (1-r2)*(nf-1)/(nf-kf-1)
}

func chiSquareCDF(x, df float64) float64 {
	if x <= 0 {
		return 0
	}
	return gammaInc(df/2, x/2)
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
