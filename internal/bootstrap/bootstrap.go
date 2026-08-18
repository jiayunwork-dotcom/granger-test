// Package bootstrap 提供自助法和置换检验。
package bootstrap

import (
	"math"
	"math/rand"
	"sort"
)

// ResidualBootstrap 残差自助法。
func ResidualBootstrap(fitted, residuals []float64, rng *rand.Rand) []float64 {
	n := len(fitted)
	sample := make([]float64, n)
	for i := 0; i < n; i++ {
		idx := rng.Intn(len(residuals))
		sample[i] = fitted[i] + residuals[idx]
	}
	return sample
}

// BlockBootstrap 块自助法（保持时间依赖结构）。
func BlockBootstrap(data []float64, blockSize int, rng *rand.Rand) []float64 {
	n := len(data)
	if blockSize <= 0 {
		blockSize = 1
	}
	sample := make([]float64, 0, n)
	for len(sample) < n {
		start := rng.Intn(n)
		for j := 0; j < blockSize && len(sample) < n; j++ {
			idx := (start + j) % n
			sample = append(sample, data[idx])
		}
	}
	return sample[:n]
}

// StationaryBootstrap Politis-Romano 平稳自助法。
func StationaryBootstrap(data []float64, prob float64, rng *rand.Rand) []float64 {
	n := len(data)
	if prob <= 0 || prob >= 1 {
		prob = 0.1
	}
	sample := make([]float64, n)
	idx := rng.Intn(n)
	for i := 0; i < n; i++ {
		sample[i] = data[idx]
		if rng.Float64() < prob {
			idx = rng.Intn(n)
		} else {
			idx = (idx + 1) % n
		}
	}
	return sample
}

// ConfidenceInterval 计算 bootstrap 统计量的置信区间。
func ConfidenceInterval(stats []float64, confidence float64) (lo, hi float64) {
	sorted := make([]float64, len(stats))
	copy(sorted, stats)
	sort.Float64s(sorted)
	alpha := (1 - confidence) / 2
	n := len(sorted)
	loIdx := int(math.Floor(alpha * float64(n)))
	hiIdx := int(math.Floor((1 - alpha) * float64(n)))
	if loIdx < 0 {
		loIdx = 0
	}
	if hiIdx >= n {
		hiIdx = n - 1
	}
	return sorted[loIdx], sorted[hiIdx]
}

// BootstrapMean 自助法估计均值的置信区间。
func BootstrapMean(data []float64, nIter int, confidence float64, rng *rand.Rand) (lo, hi float64) {
	n := len(data)
	stats := make([]float64, nIter)
	for b := 0; b < nIter; b++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += data[rng.Intn(n)]
		}
		stats[b] = sum / float64(n)
	}
	return ConfidenceInterval(stats, confidence)
}

// BootstrapVariance 自助法估计标准误。
func BootstrapVariance(data []float64, nIter int, rng *rand.Rand) float64 {
	n := len(data)
	stats := make([]float64, nIter)
	for b := 0; b < nIter; b++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += data[rng.Intn(n)]
		}
		stats[b] = sum / float64(n)
	}
	mean := 0.0
	for _, s := range stats {
		mean += s
	}
	mean /= float64(nIter)
	variance := 0.0
	for _, s := range stats {
		d := s - mean
		variance += d * d
	}
	variance /= float64(nIter - 1)
	return math.Sqrt(variance)
}

// PermutationTest 置换检验。
func PermutationTest(x, y []float64, nPerm int, rng *rand.Rand) float64 {
	nx := len(x)
	combined := make([]float64, nx+len(y))
	copy(combined, x)
	copy(combined[nx:], y)
	observedDiff := math.Abs(mean(x) - mean(y))
	count := 0
	for p := 0; p < nPerm; p++ {
		rng.Shuffle(len(combined), func(i, j int) {
			combined[i], combined[j] = combined[j], combined[i]
		})
		permDiff := math.Abs(mean(combined[:nx]) - mean(combined[nx:]))
		if permDiff >= observedDiff {
			count++
		}
	}
	return float64(count) / float64(nPerm)
}

func mean(data []float64) float64 {
	s := 0.0
	for _, v := range data {
		s += v
	}
	return s / float64(len(data))
}
