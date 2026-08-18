// Package causality 提供格兰杰因果之外的因果推断方法。
// 包括传递熵、收敛交叉映射和瞬时因果检验。
package causality

import (
	"fmt"
	"math"
)

// TransferEntropyResult 保存传递熵计算结果
type TransferEntropyResult struct {
	// TE 为从 X 到 Y 的传递熵值
	TE float64
	// Normalized 为归一化后的传递熵（0到1之间）
	Normalized float64
	// Lag 为使用的滞后阶数
	Lag int
}

// TransferEntropy 计算从序列 x 到序列 y 的传递熵。
// 传递熵衡量 x 的过去对 y 未来的信息贡献，超越 y 自身历史所能提供的信息。
// lag 指定使用的滞后阶数，bins 指定直方图的分箱数。
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

	// 离散化数据
	xd := discretize(x, bins)
	yd := discretize(y, bins)

	// 计算联合概率和条件概率
	// TE(X->Y) = H(Y_t | Y_{t-1:t-lag}) - H(Y_t | Y_{t-1:t-lag}, X_{t-1:t-lag})
	effective := n - lag
	// 统计联合频率
	jointXYpYp := make(map[[3]int]int) // (yt, yt-k, xt-k)
	jointYpYp := make(map[[2]int]int)  // (yt, yt-k)
	margYp := make(map[int]int)        // (yt-k)
	jointXpYp := make(map[[2]int]int)  // (xt-k, yt-k)

	for t := lag; t < n; t++ {
		yt := yd[t]
		ytk := yd[t-lag]
		xtk := xd[t-lag]

		jointXYpYp[[3]int{yt, ytk, xtk}]++
		jointYpYp[[2]int{yt, ytk}]++
		margYp[ytk]++
		jointXpYp[[2]int{xtk, ytk}]++
	}

	// 计算传递熵
	te := 0.0
	for key, count := range jointXYpYp {
		yt, ytk, xtk := key[0], key[1], key[2]
		pYtYpXp := float64(count) / float64(effective)
		pYtYp := float64(jointYpYp[[2]int{yt, ytk}]) / float64(effective)
		pYp := float64(margYp[ytk]) / float64(effective)
		pXpYp := float64(jointXpYp[[2]int{xtk, ytk}]) / float64(effective)

		if pYtYp > 0 && pXpYp > 0 && pYp > 0 {
			// TE = sum p(yt, yt-k, xt-k) * log( p(yt|yt-k,xt-k) / p(yt|yt-k) )
			condFull := pYtYpXp * pYp / pXpYp   // p(yt | yt-k, xt-k) ∝ p(yt,yt-k,xt-k) * p(yt-k) / p(xt-k,yt-k)
			condReduced := pYtYp / pYp            // p(yt | yt-k) ∝ p(yt,yt-k) / p(yt-k)
			if condFull > 0 && condReduced > 0 {
				te += pYtYpXp * math.Log2(condFull/condReduced)
			}
		}
	}

	// 归一化：除以 Y 的熵
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

// CCMResult 保存收敛交叉映射结果
type CCMResult struct {
	// Rho 为预测相关系数
	Rho float64
	// LibSizes 为使用的库大小序列
	LibSizes []int
	// Rhos 为对应各库大小的相关系数
	Rhos []float64
}

// ConvergentCrossMapping 实现 Sugihara 等人提出的收敛交叉映射方法。
// 用于检测非线性动态系统中的因果关系。
// embDim 为嵌入维度，tau 为时间延迟。
func ConvergentCrossMapping(x, y []float64, embDim, tau int) (*CCMResult, error) {
	n := len(x)
	if n != len(y) {
		return nil, fmt.Errorf("序列长度不一致")
	}
	minLen := (embDim-1)*tau + 1
	if n < minLen+embDim {
		return nil, fmt.Errorf("数据长度不足: 需要至少 %d 个观测值", minLen+embDim)
	}

	// 构建影子流形 (shadow manifold)
	validLen := n - (embDim-1)*tau
	manifoldX := buildManifold(x, embDim, tau, validLen)
	_ = buildManifold(y, embDim, tau, validLen)

	// 计算不同库大小下的预测能力
	libSizes := make([]int, 0)
	rhos := make([]float64, 0)
	step := validLen / 10
	if step < 1 {
		step = 1
	}

	for lib := embDim + 2; lib <= validLen; lib += step {
		// 使用 x 的流形预测 y
		predicted := crossMapPredict(manifoldX, y[(embDim-1)*tau:], lib, embDim+1)
		actual := y[(embDim-1)*tau : (embDim-1)*tau+lib]
		rho := correlation(predicted, actual)
		libSizes = append(libSizes, lib)
		rhos = append(rhos, rho)
	}

	// 最终相关系数取最大库大小的值
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

// InstantaneousCausalityResult 保存瞬时因果检验结果
type InstantaneousCausalityResult struct {
	Statistic float64
	PValue    float64
	DF        int
}

// InstantaneousCausality 检验两个序列之间的瞬时（同期）因果关系。
// 基于 VAR 模型残差的相关性进行检验。
func InstantaneousCausality(residX, residY []float64) (*InstantaneousCausalityResult, error) {
	n := len(residX)
	if n != len(residY) {
		return nil, fmt.Errorf("残差序列长度不一致")
	}
	if n < 3 {
		return nil, fmt.Errorf("样本量不足")
	}

	// 计算残差相关系数
	rho := correlation(residX, residY)

	// Fisher z 变换检验
	z := 0.5 * math.Log((1+rho)/(1-rho))
	stat := z * math.Sqrt(float64(n-3))

	// 近似 p 值（双侧检验）
	pValue := 2 * (1 - normalCDF(math.Abs(stat)))

	return &InstantaneousCausalityResult{
		Statistic: stat,
		PValue:    pValue,
		DF:        n - 3,
	}, nil
}

// buildManifold 构建时间延迟嵌入的影子流形
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

// crossMapPredict 使用流形近邻进行交叉映射预测
func crossMapPredict(manifold [][]float64, target []float64, libSize, knn int) []float64 {
	predicted := make([]float64, libSize)
	for i := 0; i < libSize; i++ {
		// 找到 knn 个最近邻并加权预测
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
				// 替换最远的邻居
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

		// 计算权重（指数衰减）
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

// euclidean 计算两个向量的欧几里得距离
func euclidean(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// correlation 计算皮尔逊相关系数
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

// normalCDF 标准正态分布的累积分布函数
func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

// discretize 将连续数据离散化为整数分箱
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
