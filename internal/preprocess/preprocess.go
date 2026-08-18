// Package preprocess 提供时间序列预处理工具。
package preprocess

import (
	"math"
	"sort"
)

// Diff 计算 d 阶差分。
func Diff(data []float64, d int) []float64 {
	result := data
	for i := 0; i < d; i++ {
		if len(result) < 2 {
			return nil
		}
		next := make([]float64, len(result)-1)
		for j := 1; j < len(result); j++ {
			next[j-1] = result[j] - result[j-1]
		}
		result = next
	}
	return result
}

// Detrend 去除线性趋势。
func Detrend(data []float64) []float64 {
	n := len(data)
	if n < 2 {
		return data
	}
	meanX := float64(n-1) / 2
	meanY := 0.0
	for _, v := range data {
		meanY += v
	}
	meanY /= float64(n)
	num := 0.0
	den := 0.0
	for i, v := range data {
		dx := float64(i) - meanX
		num += dx * (v - meanY)
		den += dx * dx
	}
	if den == 0 {
		return data
	}
	slope := num / den
	intercept := meanY - slope*meanX
	result := make([]float64, n)
	for i, v := range data {
		result[i] = v - (slope*float64(i) + intercept)
	}
	return result
}

// Standardize 标准化（均值为0，标准差为1）。
func Standardize(data []float64) []float64 {
	n := len(data)
	if n < 2 {
		return data
	}
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(n)
	std := 0.0
	for _, v := range data {
		d := v - mean
		std += d * d
	}
	std = math.Sqrt(std / float64(n-1))
	if std == 0 {
		return make([]float64, n)
	}
	result := make([]float64, n)
	for i, v := range data {
		result[i] = (v - mean) / std
	}
	return result
}

// LogTransform 对数变换（正值）。
func LogTransform(data []float64) []float64 {
	result := make([]float64, len(data))
	for i, v := range data {
		if v > 0 {
			result[i] = math.Log(v)
		}
	}
	return result
}

// FillNA 用前值填充 NaN。
func FillNA(data []float64) []float64 {
	result := make([]float64, len(data))
	last := 0.0
	for i, v := range data {
		if math.IsNaN(v) {
			result[i] = last
		} else {
			result[i] = v
			last = v
		}
	}
	return result
}

// Winsorize 缩尾处理（将超出 [p, 100-p] 百分位的值截断）。
func Winsorize(data []float64, p float64) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)
	lo := sorted[int(math.Floor(p/100*float64(n)))]
	hi := sorted[int(math.Floor((100-p)/100*float64(n-1)))]
	result := make([]float64, n)
	for i, v := range data {
		if v < lo {
			result[i] = lo
		} else if v > hi {
			result[i] = hi
		} else {
			result[i] = v
		}
	}
	return result
}

// SeasonalDiff 计算季节差分。
func SeasonalDiff(data []float64, period int) []float64 {
	if period <= 0 || period >= len(data) {
		return nil
	}
	result := make([]float64, len(data)-period)
	for i := period; i < len(data); i++ {
		result[i-period] = data[i] - data[i-period]
	}
	return result
}

// MovingMedian 移动中位数。
func MovingMedian(data []float64, window int) []float64 {
	n := len(data)
	if n == 0 || window < 1 {
		return nil
	}
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		start := i - window/2
		end := i + window/2 + 1
		if start < 0 {
			start = 0
		}
		if end > n {
			end = n
		}
		win := make([]float64, end-start)
		copy(win, data[start:end])
		sort.Float64s(win)
		result[i] = win[len(win)/2]
	}
	return result
}

// BoxCox Box-Cox 变换。
func BoxCox(data []float64, lambda float64) []float64 {
	result := make([]float64, len(data))
	for i, v := range data {
		if v <= 0 {
			result[i] = 0
			continue
		}
		if math.Abs(lambda) < 1e-10 {
			result[i] = math.Log(v)
		} else {
			result[i] = (math.Pow(v, lambda) - 1) / lambda
		}
	}
	return result
}
