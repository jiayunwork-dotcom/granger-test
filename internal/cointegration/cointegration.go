// Package cointegration 提供协整检验与向量误差修正模型（VECM）。
// 包括 Engle-Granger 两步法、Johansen 迹检验和 VECM 基本结构。
package cointegration

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

// EngleGrangerResult 保存 Engle-Granger 两步法检验结果
type EngleGrangerResult struct {
	// CointCoeff 为协整向量（归一化后第一个变量系数为1）
	CointCoeff []float64
	// ADFStat 为残差的 ADF 统计量
	ADFStat float64
	// PValue 为 ADF 检验的 p 值
	PValue float64
	// Residuals 为协整回归残差
	Residuals []float64
	// Cointegrated 是否存在协整关系
	Cointegrated bool
}

// JohansenResult 保存 Johansen 迹检验结果
type JohansenResult struct {
	// TraceStats 为各协整秩对应的迹统计量
	TraceStats []float64
	// CriticalValues 为 5% 显著性水平的临界值
	CriticalValues []float64
	// Rank 为估计的协整秩
	Rank int
	// Eigenvectors 为特征向量（协整向量）
	Eigenvectors [][]float64
}

// VECMResult 保存向量误差修正模型结果
type VECMResult struct {
	// Alpha 为调整系数矩阵
	Alpha [][]float64
	// Beta 为协整向量矩阵
	Beta [][]float64
	// Gamma 为短期动态系数
	Gamma [][][]float64
	// Residuals 为模型残差
	Residuals [][]float64
}

// EngleGranger 执行 Engle-Granger 两步法协整检验。
// 第一步：对序列进行 OLS 回归；第二步：对残差进行 ADF 检验。
// y 为因变量序列，x 为自变量序列矩阵的列（每列一个变量）。
func EngleGranger(y []float64, x [][]float64) (*EngleGrangerResult, error) {
	n := len(y)
	if n == 0 {
		return nil, fmt.Errorf("因变量序列为空")
	}
	numVars := len(x)
	if numVars == 0 {
		return nil, fmt.Errorf("自变量序列为空")
	}
	for i, xi := range x {
		if len(xi) != n {
			return nil, fmt.Errorf("第 %d 个自变量长度 %d 与因变量长度 %d 不一致", i, len(xi), n)
		}
	}

	// 第一步：协整回归 y = β0 + β1*x1 + β2*x2 + ... + e
	k := numVars + 1 // 含截距
	X := make([][]float64, n)
	for t := 0; t < n; t++ {
		row := make([]float64, k)
		row[0] = 1.0 // 截距
		for j := 0; j < numVars; j++ {
			row[j+1] = x[j][t]
		}
		X[t] = row
	}

	beta, _, err := ols.Fit(X, y)
	if err != nil {
		return nil, fmt.Errorf("协整回归失败: %w", err)
	}

	// 计算残差
	residuals := make([]float64, n)
	for t := 0; t < n; t++ {
		pred := beta[0]
		for j := 0; j < numVars; j++ {
			pred += beta[j+1] * x[j][t]
		}
		residuals[t] = y[t] - pred
	}

	// 第二步：对残差进行 ADF 检验
	adfStat, pValue := adfOnResiduals(residuals)

	return &EngleGrangerResult{
		CointCoeff:   beta,
		ADFStat:      adfStat,
		PValue:       pValue,
		Residuals:    residuals,
		Cointegrated: pValue < 0.05,
	}, nil
}

// JohansenTrace 执行 Johansen 迹检验。
// data 为多变量时间序列（每列一个变量），lagOrder 为 VAR 滞后阶数。
// 返回各协整秩假设的检验结果。
func JohansenTrace(data [][]float64, lagOrder int) (*JohansenResult, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, fmt.Errorf("输入数据为空")
	}
	n := len(data)    // 观测数
	k := len(data[0]) // 变量数
	if lagOrder <= 0 || lagOrder >= n/2 {
		return nil, fmt.Errorf("无效的滞后阶数: %d", lagOrder)
	}

	// 计算一阶差分
	T := n - lagOrder - 1
	if T < k+1 {
		return nil, fmt.Errorf("样本量不足")
	}

	// 构建差分序列和水平序列
	dY := make([][]float64, T)
	Ylevel := make([][]float64, T)
	for t := 0; t < T; t++ {
		idx := t + lagOrder
		dY[t] = make([]float64, k)
		Ylevel[t] = make([]float64, k)
		for j := 0; j < k; j++ {
			dY[t][j] = data[idx+1][j] - data[idx][j]
			Ylevel[t][j] = data[idx][j]
		}
	}

	// 计算残差矩阵的特征值（简化实现）
	// 计算 S00、S01、S11 矩阵
	S00 := covMatrix(dY, k)
	S11 := covMatrix(Ylevel, k)
	S01 := crossCovMatrix(dY, Ylevel, k)

	// 求解广义特征值问题的近似
	eigenvalues := solveEigenApprox(S00, S01, S11, k)

	// 计算迹统计量
	traceStats := make([]float64, k)
	for r := 0; r < k; r++ {
		stat := 0.0
		for i := r; i < k; i++ {
			if i < len(eigenvalues) && eigenvalues[i] > 0 && eigenvalues[i] < 1 {
				stat -= float64(T) * math.Log(1-eigenvalues[i])
			}
		}
		traceStats[r] = stat
	}

	// 简化的临界值（5% 显著性水平，基于 Osterwald-Lenum 表）
	critValues := johansenCriticalValues(k)

	// 确定协整秩
	rank := 0
	for r := 0; r < k; r++ {
		if r < len(critValues) && traceStats[r] > critValues[r] {
			rank = r + 1
		}
	}

	return &JohansenResult{
		TraceStats:     traceStats,
		CriticalValues: critValues,
		Rank:           rank,
	}, nil
}

// VECM 估计向量误差修正模型。
// data 为多变量时间序列，rank 为协整秩，lagOrder 为差分项滞后阶数。
func VECM(data [][]float64, rank, lagOrder int) (*VECMResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("输入数据为空")
	}
	n := len(data)
	k := len(data[0])
	if rank <= 0 || rank > k {
		return nil, fmt.Errorf("无效的协整秩: %d（变量数: %d）", rank, k)
	}
	if lagOrder <= 0 {
		return nil, fmt.Errorf("滞后阶数必须为正整数")
	}

	T := n - lagOrder - 1
	if T < 2*(k*lagOrder+rank) {
		return nil, fmt.Errorf("样本量不足以估计 VECM")
	}

	// 构建差分因变量
	dY := make([][]float64, T)
	for t := 0; t < T; t++ {
		idx := t + lagOrder
		dY[t] = make([]float64, k)
		for j := 0; j < k; j++ {
			dY[t][j] = data[idx+1][j] - data[idx][j]
		}
	}

	// 构建误差修正项（使用水平序列的前 rank 个主成分近似）
	// 简化处理：使用第一个 rank 变量作为误差修正项
	ect := make([][]float64, T)
	for t := 0; t < T; t++ {
		idx := t + lagOrder
		ect[t] = make([]float64, rank)
		for r := 0; r < rank; r++ {
			ect[t][r] = data[idx][r]
		}
	}

	// 对每个方程单独估计
	alpha := make([][]float64, k)
	residuals := make([][]float64, T)
	for t := 0; t < T; t++ {
		residuals[t] = make([]float64, k)
	}

	for eq := 0; eq < k; eq++ {
		// 设计矩阵：截距 + ECT + 滞后差分
		numRegressors := 1 + rank + k*lagOrder
		X := make([][]float64, T)
		yEq := make([]float64, T)

		for t := 0; t < T; t++ {
			row := make([]float64, numRegressors)
			col := 0
			row[col] = 1.0 // 截距
			col++
			// 误差修正项
			for r := 0; r < rank; r++ {
				row[col] = ect[t][r]
				col++
			}
			// 滞后差分项
			for lag := 1; lag <= lagOrder; lag++ {
				idx := t + lagOrder - lag
				if idx >= 0 && idx < n-1 {
					for j := 0; j < k; j++ {
						row[col] = data[idx+1][j] - data[idx][j]
						col++
					}
				} else {
					col += k
				}
			}
			X[t] = row
			yEq[t] = dY[t][eq]
		}

		beta, _, err := ols.Fit(X, yEq)
		if err != nil {
			alpha[eq] = make([]float64, rank)
			continue
		}

		// 提取调整系数
		alpha[eq] = make([]float64, rank)
		for r := 0; r < rank; r++ {
			if r+1 < len(beta) {
				alpha[eq][r] = beta[r+1]
			}
		}

		// 计算残差
		for t := 0; t < T; t++ {
			pred := 0.0
			for j, b := range beta {
				if j < len(X[t]) {
					pred += b * X[t][j]
				}
			}
			residuals[t][eq] = yEq[t] - pred
		}
	}

	return &VECMResult{
		Alpha:     alpha,
		Residuals: residuals,
	}, nil
}

// adfOnResiduals 对残差执行简化的 ADF 检验
func adfOnResiduals(residuals []float64) (float64, float64) {
	n := len(residuals)
	if n < 4 {
		return 0, 1
	}

	// ΔY_t = γ * Y_{t-1} + ε_t（不含截距，因为残差均值为零）
	T := n - 1
	X := make([][]float64, T)
	y := make([]float64, T)
	for t := 0; t < T; t++ {
		X[t] = []float64{residuals[t]}
		y[t] = residuals[t+1] - residuals[t]
	}

	beta, rss, err := ols.Fit(X, y)
	if err != nil || len(beta) == 0 {
		return 0, 1
	}

	// t 统计量
	gamma := beta[0]
	se := math.Sqrt(rss / float64(T-1))
	// 计算 X'X 的逆对角元
	sumX2 := 0.0
	for t := 0; t < T; t++ {
		sumX2 += X[t][0] * X[t][0]
	}
	if sumX2 == 0 {
		return 0, 1
	}
	seGamma := se / math.Sqrt(sumX2)
	if seGamma == 0 {
		return 0, 1
	}
	tStat := gamma / seGamma

	// 使用 MacKinnon 近似临界值（协整残差）
	// 近似 p 值
	pValue := adfPValueApprox(tStat, n)

	return tStat, pValue
}

// adfPValueApprox 近似 ADF 检验 p 值（基于渐近分布）
func adfPValueApprox(tStat float64, n int) float64 {
	// 简化的近似：使用正态分布的尾部概率并调整
	// 实际 ADF 分布偏左，临界值约 -3.34（5%）用于协整残差
	// 这里使用线性插值近似
	if tStat < -4.0 {
		return 0.01
	}
	if tStat < -3.34 {
		return 0.05
	}
	if tStat < -2.86 {
		return 0.10
	}
	if tStat < -2.57 {
		return 0.15
	}
	// p > 0.15 时不太可能拒绝
	return 0.5 + 0.5*normalCDF(tStat)
}

// covMatrix 计算样本协方差矩阵
func covMatrix(data [][]float64, k int) [][]float64 {
	n := len(data)
	means := make([]float64, k)
	for _, row := range data {
		for j := 0; j < k; j++ {
			means[j] += row[j]
		}
	}
	for j := range means {
		means[j] /= float64(n)
	}

	S := make([][]float64, k)
	for i := 0; i < k; i++ {
		S[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			sum := 0.0
			for t := 0; t < n; t++ {
				sum += (data[t][i] - means[i]) * (data[t][j] - means[j])
			}
			S[i][j] = sum / float64(n-1)
		}
	}
	return S
}

// crossCovMatrix 计算两组数据的交叉协方差矩阵
func crossCovMatrix(a, b [][]float64, k int) [][]float64 {
	n := len(a)
	meansA := make([]float64, k)
	meansB := make([]float64, k)
	for t := 0; t < n; t++ {
		for j := 0; j < k; j++ {
			meansA[j] += a[t][j]
			meansB[j] += b[t][j]
		}
	}
	for j := range meansA {
		meansA[j] /= float64(n)
		meansB[j] /= float64(n)
	}

	S := make([][]float64, k)
	for i := 0; i < k; i++ {
		S[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			sum := 0.0
			for t := 0; t < n; t++ {
				sum += (a[t][i] - meansA[i]) * (b[t][j] - meansB[j])
			}
			S[i][j] = sum / float64(n-1)
		}
	}
	return S
}

// solveEigenApprox 近似求解广义特征值问题
// 用于 Johansen 检验中求解 |λS11 - S01'inv(S00)S01| = 0
func solveEigenApprox(S00, S01, S11 [][]float64, k int) []float64 {
	// 简化实现：计算 inv(S11)*S01'*inv(S00)*S01 的对角元作为特征值近似
	eigenvalues := make([]float64, k)
	for i := 0; i < k; i++ {
		// 使用对角元比值的简化近似
		if S11[i][i] > 0 && S00[i][i] > 0 {
			ratio := S01[i][i] * S01[i][i] / (S00[i][i] * S11[i][i])
			if ratio > 1 {
				ratio = 0.99
			}
			if ratio < 0 {
				ratio = 0
			}
			eigenvalues[i] = ratio
		}
	}
	// 按降序排列
	for i := 0; i < k-1; i++ {
		for j := i + 1; j < k; j++ {
			if eigenvalues[j] > eigenvalues[i] {
				eigenvalues[i], eigenvalues[j] = eigenvalues[j], eigenvalues[i]
			}
		}
	}
	return eigenvalues
}

// johansenCriticalValues 返回 Johansen 迹检验的 5% 临界值
// 基于变量数 k 的常用临界值表
func johansenCriticalValues(k int) []float64 {
	// 简化的临界值表（5% 显著性水平）
	tables := map[int][]float64{
		1: {3.84},
		2: {15.41, 3.76},
		3: {29.68, 15.41, 3.76},
		4: {47.21, 29.68, 15.41, 3.76},
		5: {68.52, 47.21, 29.68, 15.41, 3.76},
	}
	if cv, ok := tables[k]; ok {
		return cv
	}
	// 对于更大的 k，使用线性外推
	cv := make([]float64, k)
	for i := 0; i < k; i++ {
		cv[i] = float64(k-i) * 15.0
	}
	return cv
}

// normalCDF 标准正态累积分布函数
func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}
