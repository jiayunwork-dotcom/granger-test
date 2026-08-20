// Package spectral 提供频谱分析工具。
package spectral

import "math"

// Complex 复数。
type Complex struct{ Re, Im float64 }

// DFT 离散傅里叶变换。
func DFT(x []float64) []Complex {
	n := len(x)
	result := make([]Complex, n)
	for k := 0; k < n; k++ {
		for t := 0; t < n; t++ {
			angle := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			result[k].Re += x[t] * math.Cos(angle)
			result[k].Im += x[t] * math.Sin(angle)
		}
	}
	return result
}

// PowerSpectrum 计算功率谱密度。
func PowerSpectrum(x []float64) []float64 {
	dft := DFT(x)
	n := len(dft)
	psd := make([]float64, n/2+1)
	for k := 0; k <= n/2; k++ {
		psd[k] = (dft[k].Re*dft[k].Re + dft[k].Im*dft[k].Im) / float64(n)
	}
	return psd
}

// Periodogram 计算周期图。
func Periodogram(x []float64) []float64 {
	n := len(x)
	if n == 0 {
		return nil
	}
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)
	centered := make([]float64, n)
	for i, v := range x {
		centered[i] = v - mean
	}
	return PowerSpectrum(centered)
}

// DominantFreq 返回功率谱中的主频率（除DC分量外）。
func DominantFreq(psd []float64) int {
	if len(psd) < 2 {
		return 0
	}
	maxIdx := 1
	maxVal := psd[1]
	for i := 2; i < len(psd); i++ {
		if psd[i] > maxVal {
			maxVal = psd[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// Coherence 计算两个信号的相干性。
func Coherence(x, y []float64) []float64 {
	n := len(x)
	if n == 0 || len(y) != n {
		return nil
	}
	dftX := DFT(x)
	dftY := DFT(y)
	coh := make([]float64, n/2+1)
	for k := 0; k <= n/2; k++ {
		// 交叉功率谱
		crossRe := dftX[k].Re*dftY[k].Re + dftX[k].Im*dftY[k].Im
		crossIm := dftX[k].Im*dftY[k].Re - dftX[k].Re*dftY[k].Im
		crossPow := crossRe*crossRe + crossIm*crossIm
		pxxk := dftX[k].Re*dftX[k].Re + dftX[k].Im*dftX[k].Im
		pyyk := dftY[k].Re*dftY[k].Re + dftY[k].Im*dftY[k].Im
		denom := bandDenom(pxxk, pyyk)
		if denom > 0 {
			coh[k] = crossPow / denom
		}
	}
	return coh
}

// SpectralDensity 使用 Welch 方法估计功率谱。
func SpectralDensity(x []float64, segLen int) []float64 {
	n := len(x)
	if segLen <= 0 || segLen > n {
		segLen = n
	}
	overlap := segLen / 2
	nSegs := 0
	var avgPSD []float64
	for start := 0; start+segLen <= n; start += (segLen - overlap) {
		seg := x[start : start+segLen]
		windowed := Hann(seg)
		psd := PowerSpectrum(windowed)
		if avgPSD == nil {
			avgPSD = make([]float64, len(psd))
		}
		for i := range psd {
			avgPSD[i] += psd[i]
		}
		nSegs++
	}
	if nSegs == 0 {
		return nil
	}
	for i := range avgPSD {
		avgPSD[i] /= float64(nSegs)
	}
	return avgPSD
}

// Hann 应用 Hann 窗。
func Hann(x []float64) []float64 {
	n := len(x)
	result := make([]float64, n)
	for i, v := range x {
		w := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		result[i] = v * w
	}
	return result
}
