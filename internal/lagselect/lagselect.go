// Package lagselect 提供 VAR 模型的滞后阶数选择方法。
// 包括 FPE（最终预测误差）准则、LR 序贯检验和综合最优滞后选择。
package lagselect

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

// CriterionResult 保存单个信息准则的计算结果
type CriterionResult struct {
	Lag   int
	Value float64
}

// LagSelectionResult 保存综合滞后选择结果
type LagSelectionResult struct {
	OptimalLag int
	FPE        []CriterionResult
	AIC        []CriterionResult
	BIC        []CriterionResult
	LRTest     []LRTestResult
}

// LRTestResult 保存似然比检验结果
type LRTestResult struct {
	Lag       int
	Statistic float64
	PValue    float64
	Reject    bool
}

// FPE 计算最终预测误差（Final Prediction Error）准则。
// FPE(p) = ((T+k)/(T-k)) * (RSS/T)
// 其中 T 为样本量，k 为参数个数，RSS 为残差平方和。
func FPE(y []float64, maxLag int) ([]CriterionResult, int, error) {
	n := len(y)
	if maxLag <= 0 || maxLag >= n/2 {
		return nil, 0, fmt.Errorf("无效的最大滞后阶数: %d (样本量: %d)", maxLag, n)
	}

	results := make([]CriterionResult, 0, maxLag)
	bestLag := 1
	bestFPE := math.Inf(1)

	for p := 1; p <= maxLag; p++ {
		// 构建设计矩阵：含截距和 p 阶滞后
		T := n - p
		k := p + 1 // 滞后变量 + 截距

		X := make([][]float64, T)
		yDep := make([]float64, T)
		for t := 0; t < T; t++ {
			row := make([]float64, k)
			row[0] = 1.0 // 截距
			for j := 1; j <= p; j++ {
				row[j] = y[t+p-j]
			}
			X[t] = row
			yDep[t] = y[t+p]
		}

		_, rss, err := ols.Fit(X, yDep)
		if err != nil {
			continue
		}

		// 计算 FPE
		fpe := (float64(T+k) / float64(T-k)) * (rss / float64(T))
		results = append(results, CriterionResult{Lag: p, Value: fpe})

		if fpe < bestFPE {
			bestFPE = fpe
			bestLag = p
		}
	}

	if len(results) == 0 {
		return nil, 0, fmt.Errorf("无法计算任何滞后阶数的 FPE")
	}
	return results, bestLag, nil
}

// LRSequential 执行序贯似然比检验选择最优滞后阶数。
// 从 maxLag 开始逐步减少，检验零假设 H0: 第 p 阶滞后系数为零。
// significance 为显著性水平（通常 0.05）。
func LRSequential(y []float64, maxLag int, significance float64) ([]LRTestResult, int, error) {
	n := len(y)
	if maxLag <= 1 || maxLag >= n/2 {
		return nil, 0, fmt.Errorf("无效的最大滞后阶数")
	}

	results := make([]LRTestResult, 0)
	selectedLag := maxLag

	// 从最大滞后开始向下检验
	for p := maxLag; p >= 2; p-- {
		// 无约束模型 RSS（p 阶）
		rssU, TU, err := fitAR(y, p)
		if err != nil {
			continue
		}
		// 约束模型 RSS（p-1 阶）
		rssR, _, err := fitAR(y, p-1)
		if err != nil {
			continue
		}

		// LR 统计量 = T * log(RSS_R / RSS_U)
		lr := float64(TU) * math.Log(rssR/rssU)
		// 自由度为约束数（此处为1）
		df := 1
		pValue := 1 - chiSquaredCDF(lr, df)

		reject := pValue < significance
		results = append(results, LRTestResult{
			Lag:       p,
			Statistic: lr,
			PValue:    pValue,
			Reject:    reject,
		})

		if reject {
			selectedLag = p
			break
		}
		selectedLag = p - 1
	}

	return results, selectedLag, nil
}

// OptimalLag 综合多种准则选择最优滞后阶数。
// 结合 FPE、AIC、BIC 和 LR 检验的结果，采用多数投票法。
func OptimalLag(y []float64, maxLag int) (*LagSelectionResult, error) {
	n := len(y)
	if maxLag <= 0 || maxLag >= n/2 {
		return nil, fmt.Errorf("无效的最大滞后阶数: %d", maxLag)
	}

	result := &LagSelectionResult{}

	// 计算各准则
	fpeResults, fpeBest, _ := FPE(y, maxLag)
	result.FPE = fpeResults

	// 计算 AIC 和 BIC
	aicResults := make([]CriterionResult, 0, maxLag)
	bicResults := make([]CriterionResult, 0, maxLag)
	aicBest, bicBest := 1, 1
	aicMin, bicMin := math.Inf(1), math.Inf(1)

	for p := 1; p <= maxLag; p++ {
		rss, T, err := fitAR(y, p)
		if err != nil {
			continue
		}
		k := float64(p + 1)
		Tf := float64(T)
		logL := -Tf/2*math.Log(2*math.Pi) - Tf/2*math.Log(rss/Tf) - Tf/2

		aic := -2*logL + 2*k
		bic := -2*logL + k*math.Log(Tf)

		aicResults = append(aicResults, CriterionResult{Lag: p, Value: aic})
		bicResults = append(bicResults, CriterionResult{Lag: p, Value: bic})

		if aic < aicMin {
			aicMin = aic
			aicBest = p
		}
		if bic < bicMin {
			bicMin = bic
			bicBest = p
		}
	}
	result.AIC = aicResults
	result.BIC = bicResults

	// LR 检验
	lrResults, lrBest, _ := LRSequential(y, maxLag, 0.05)
	result.LRTest = lrResults

	// 多数投票
	votes := make(map[int]int)
	votes[fpeBest]++
	votes[aicBest]++
	votes[bicBest]++
	votes[lrBest]++

	bestLag := 1
	maxVotes := 0
	for lag, v := range votes {
		if v > maxVotes || (v == maxVotes && lag < bestLag) {
			maxVotes = v
			bestLag = lag
		}
	}
	result.OptimalLag = bestLag

	return result, nil
}

// fitAR 拟合 AR(p) 模型并返回残差平方和与有效样本量
func fitAR(y []float64, p int) (float64, int, error) {
	n := len(y)
	T := n - p
	if T <= p+1 {
		return 0, 0, fmt.Errorf("样本不足")
	}

	X := make([][]float64, T)
	yDep := make([]float64, T)
	for t := 0; t < T; t++ {
		row := make([]float64, p+1)
		row[0] = 1.0
		for j := 1; j <= p; j++ {
			row[j] = y[t+p-j]
		}
		X[t] = row
		yDep[t] = y[t+p]
	}

	_, rss, err := ols.Fit(X, yDep)
	return rss, T, err
}

// chiSquaredCDF 卡方分布的近似 CDF（使用 Wilson-Hilferty 近似）
func chiSquaredCDF(x float64, df int) float64 {
	if x <= 0 {
		return 0
	}
	k := float64(df)
	// Wilson-Hilferty 近似
	z := math.Pow(x/k, 1.0/3.0) - (1 - 2.0/(9*k))
	z /= math.Sqrt(2.0 / (9 * k))
	return normalCDF(z)
}

// normalCDF 标准正态累积分布函数
func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}
