package causality

import (
	"fmt"
	"math"
)

type TransferEntropyResult struct {
	TE         float64
	Normalized float64
	Lag        int
}

func TransferEntropy(x, y []float64, lag, bins int) (*TransferEntropyResult, error) {
	n := len(x)
	if n != len(y) {
		return nil, fmt.Errorf("序列长度不一致: x=%d, y=%d", n, len(y))
	}
	if n <= lag {
		return nil, fmt.Errorf("数据长度 %d 不足以支持滞后 %d", n, lag)
	}
	if bins <= 0 {
		bins = 10
	}

	xd := discretize(x, bins)
	yd := discretize(y, bins)

	effective := n - lag
	jointXYpYp := make(map[[3]int]int)
	jointYpYp := make(map[[2]int]int)
	margYp := make(map[int]int)
	jointXpYp := make(map[[2]int]int)

	for t := lag; t < n; t++ {
		yt := yd[t]
		ytk := yd[t-lag]
		xtk := xd[t-lag]

		jointXYpYp[[3]int{yt, ytk, xtk}]++
		jointYpYp[[2]int{yt, ytk}]++
		margYp[ytk]++
		jointXpYp[[2]int{xtk, ytk}]++
	}

	te := 0.0
	for key, count := range jointXYpYp {
		yt, ytk, xtk := key[0], key[1], key[2]
		pYtYpXp := float64(count) / float64(effective)
		pYtYp := float64(jointYpYp[[2]int{yt, ytk}]) / float64(effective)
		pYp := float64(margYp[ytk]) / float64(effective)
		pXpYp := float64(jointXpYp[[2]int{xtk, ytk}]) / float64(effective)

		if pYtYp > 0 && pXpYp > 0 && pYp > 0 {
			condFull := pYtYpXp * pYp / pXpYp
			condReduced := pYtYp / pYp
			if condFull > 0 && condReduced > 0 {
				te += pYtYpXp * math.Log2(condFull/condReduced)
			}
		}
	}

	hy := 0.0
	for _, c := range margYp {
		p := float64(c) / float64(effective)
		if p > 0 {
			hy -= p * math.Log2(p)
		}
	}
	normalized := 0.0
	if hy > 0 {
		normalized = math.Abs(te) / hy
	}
	if normalized > 1 {
		normalized = 1
	}

	return &TransferEntropyResult{
		TE:         te,
		Normalized: normalized,
		Lag:        lag,
	}, nil
}

type CCMResult struct {
	Rho      float64
	LibSizes []int
	Rhos     []float64
}

func ConvergentCrossMapping(x, y []float64, embDim, tau int) (*CCMResult, error) {
	n := len(x)
	if n != len(y) {
		return nil, fmt.Errorf("序列长度不一致")
	}
	minLen := (embDim-1)*tau + 1
	if n < minLen+embDim {
		return nil, fmt.Errorf("数据长度不足: 需要至少 %d 个观测值", minLen+embDim)
	}

	validLen := n - (embDim-1)*tau
	manifoldX := buildManifold(x, embDim, tau, validLen)
	_ = buildManifold(y, embDim, tau, validLen)

	libSizes := make([]int, 0)
	rhos := make([]float64, 0)
	step := validLen / 10
	if step < 1 {
		step = 1
	}

	for lib := embDim + 2; lib <= validLen; lib += step {
		predicted := crossMapPredict(manifoldX, y[(embDim-1)*tau:], lib, embDim+1)
		actual := y[(embDim-1)*tau : (embDim-1)*tau+lib]
		rho := correlation(predicted, actual)
		libSizes = append(libSizes, lib)
		rhos = append(rhos, rho)
	}

	finalRho := 0.0
	if len(rhos) > 0 {
		finalRho = rhos[len(rhos)-1]
	}

	return &CCMResult{
		Rho:      finalRho,
		LibSizes: libSizes,
		Rhos:     rhos,
	}, nil
}

type InstantaneousCausalityResult struct {
	Statistic float64
	PValue    float64
	DF        int
}

func InstantaneousCausality(residX, residY []float64) (*InstantaneousCausalityResult, error) {
	n := len(residX)
	if n != len(residY) {
		return nil, fmt.Errorf("残差序列长度不一致")
	}
	if n < 3 {
		return nil, fmt.Errorf("样本量不足")
	}

	rho := correlation(residX, residY)

	z := 0.5 * math.Log((1+rho)/(1-rho))
	stat := z * math.Sqrt(float64(n-3))

	pValue := 2 * (1 - normalCDF(math.Abs(stat)))

	return &InstantaneousCausalityResult{
		Statistic: stat,
		PValue:    pValue,
		DF:        n - 3,
	}, nil
}

func buildManifold(data []float64, embDim, tau, validLen int) [][]float64 {
	manifold := make([][]float64, validLen)
	for i := 0; i < validLen; i++ {
		point := make([]float64, embDim)
		for d := 0; d < embDim; d++ {
			point[d] = data[i+d*tau]
		}
		manifold[i] = point
	}
	return manifold
}

func crossMapPredict(manifold [][]float64, target []float64, libSize, knn int) []float64 {
	predicted := make([]float64, libSize)
	for i := 0; i < libSize; i++ {
		weights := make([]float64, 0, knn)
		indices := make([]int, 0, knn)
		dists := make([]float64, 0, knn)

		for j := 0; j < libSize; j++ {
			if j == i {
				continue
			}
			d := euclidean(manifold[i], manifold[j])
			if len(dists) < knn {
				dists = append(dists, d)
				indices = append(indices, j)
			} else {
				maxIdx := 0
				for k := 1; k < len(dists); k++ {
					if dists[k] > dists[maxIdx] {
						maxIdx = k
					}
				}
				if d < dists[maxIdx] {
					dists[maxIdx] = d
					indices[maxIdx] = j
				}
			}
		}

		minDist := dists[0]
		for _, d := range dists {
			if d < minDist {
				minDist = d
			}
		}
		if minDist == 0 {
			minDist = 1e-10
		}

		totalW := 0.0
		weights = make([]float64, len(dists))
		for k, d := range dists {
			weights[k] = math.Exp(-d / minDist)
			totalW += weights[k]
		}

		pred := 0.0
		for k, idx := range indices {
			if idx < len(target) {
				pred += (weights[k] / totalW) * target[idx]
			}
		}
		predicted[i] = pred
	}
	return predicted
}

func euclidean(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func correlation(x, y []float64) float64 {
	n := len(x)
	if n == 0 || n != len(y) {
		return 0
	}
	mx, my := 0.0, 0.0
	for i := 0; i < n; i++ {
		mx += x[i]
		my += y[i]
	}
	mx /= float64(n)
	my /= float64(n)

	var num, dx, dy float64
	for i := 0; i < n; i++ {
		xi := x[i] - mx
		yi := y[i] - my
		num += xi * yi
		dx += xi * xi
		dy += yi * yi
	}
	denom := math.Sqrt(dx * dy)
	if denom == 0 {
		return 0
	}
	return num / denom
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

func discretize(data []float64, bins int) []int {
	n := len(data)
	if n == 0 {
		return nil
	}
	min, max := data[0], data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	rng := max - min
	if rng == 0 {
		result := make([]int, n)
		return result
	}
	result := make([]int, n)
	for i, v := range data {
		bin := int((v - min) / rng * float64(bins))
		if bin >= bins {
			bin = bins - 1
		}
		result[i] = bin
	}
	return result
}
