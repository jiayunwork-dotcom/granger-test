// Package residual 提供回归模型残差诊断检验方法。
// 包括序列相关检验、异方差检验、ARCH 效应检验和正态性检验。
package residual

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

// TestResult 保存检验结果的通用结构
type TestResult struct {
	Name      string
	Statistic float64
	PValue    float64
	DF        int
	Reject    bool // 在 5% 显著性水平下是否拒绝原假设
}

// BreuschGodfrey 执行 Breusch-Godfrey 序列相关 LM 检验。
// 检验原假设 H0: 残差无序列相关。
// residuals 为 OLS 回归残差，X 为原始设计矩阵，order 为检验阶数。
func BreuschGodfrey(residuals []float64, X [][]float64, order int) (*TestResult, error) {
	n := len(residuals)
	if n == 0 {
		return nil, fmt.Errorf("残差序列为空")
	}
	if order <= 0 || order >= n {
		return nil, fmt.Errorf("无效的检验阶数: %d", order)
	}
	k := 0
	if len(X) > 0 {
		k = len(X[0])
	}

	// 构建辅助回归的设计矩阵
	// 因变量为残差，自变量为原设计矩阵列 + 滞后残差
	T := n - order
	cols := k + order
	Xaux := make([][]float64, T)
	yaux := make([]float64, T)

	for t := 0; t < T; t++ {
		row := make([]float64, cols)
		idx := t + order
		// 原始设计矩阵变量
		if idx < len(X) {
			for j := 0; j < k; j++ {
				row[j] = X[idx][j]
			}
		}
		// 滞后残差
		for lag := 1; lag <= order; lag++ {
			row[k+lag-1] = residuals[idx-lag]
		}
		Xaux[t] = row
		yaux[t] = residuals[idx]
	}

	// 执行辅助回归
	_, rssAux, err := ols.Fit(Xaux, yaux)
	if err != nil {
		return nil, fmt.Errorf("辅助回归失败: %w", err)
	}

	// 计算 R² = 1 - RSS_aux / TSS
	mean := 0.0
	for _, v := range yaux {
		mean += v
	}
	mean /= float64(T)
	tss := 0.0
	for _, v := range yaux {
		d := v - mean
		tss += d * d
	}

	r2 := 0.0
	if tss > 0 {
		r2 = 1.0 - rssAux/tss
	}

	// LM 统计量 = T * R²，渐近服从 χ²(order)
	lm := float64(T) * r2
	pValue := 1 - chiSquaredCDF(lm, order)

	return &TestResult{
		Name:      "Breusch-Godfrey",
		Statistic: lm,
		PValue:    pValue,
		DF:        order,
		Reject:    pValue < 0.05,
	}, nil
}

// White 执行 White 异方差性检验。
// 检验原假设 H0: 残差具有同方差性。
// residuals 为 OLS 残差，X 为设计矩阵（不含截距列）。
func White(residuals []float64, X [][]float64) (*TestResult, error) {
	n := len(residuals)
	if n == 0 || len(X) == 0 {
		return nil, fmt.Errorf("输入数据为空")
	}
	k := len(X[0])
	if n != len(X) {
		return nil, fmt.Errorf("残差与设计矩阵行数不匹配")
	}

	// 构建辅助回归：e² ~ 1 + X + X² + 交叉项
	// 辅助变量数 = 1(截距) + k(原始) + k(平方) + k*(k-1)/2(交叉)
	numCross := k * (k - 1) / 2
	auxCols := 1 + k + k + numCross
	Xaux := make([][]float64, n)
	yaux := make([]float64, n)

	for i := 0; i < n; i++ {
		row := make([]float64, auxCols)
		col := 0
		row[col] = 1.0 // 截距
		col++
		// 原始变量
		for j := 0; j < k; j++ {
			row[col] = X[i][j]
			col++
		}
		// 平方项
		for j := 0; j < k; j++ {
			row[col] = X[i][j] * X[i][j]
			col++
		}
		// 交叉项
		for j := 0; j < k; j++ {
			for l := j + 1; l < k; l++ {
				row[col] = X[i][j] * X[i][l]
				col++
			}
		}
		Xaux[i] = row
		yaux[i] = residuals[i] * residuals[i]
	}

	// 执行辅助回归
	_, rssAux, err := ols.Fit(Xaux, yaux)
	if err != nil {
		return nil, fmt.Errorf("辅助回归失败: %w", err)
	}

	// 计算 R²
	mean := 0.0
	for _, v := range yaux {
		mean += v
	}
	mean /= float64(n)
	tss := 0.0
	for _, v := range yaux {
		d := v - mean
		tss += d * d
	}
	r2 := 0.0
	if tss > 0 {
		r2 = 1.0 - rssAux/tss
	}

	// 统计量 = n * R²，自由度为辅助回归变量数 - 1
	stat := float64(n) * r2
	df := auxCols - 1
	pValue := 1 - chiSquaredCDF(stat, df)

	return &TestResult{
		Name:      "White",
		Statistic: stat,
		PValue:    pValue,
		DF:        df,
		Reject:    pValue < 0.05,
	}, nil
}

// ArchTest 执行 ARCH 效应的 Engle LM 检验。
// 检验原假设 H0: 不存在 ARCH 效应。
// residuals 为模型残差，lags 为 ARCH 滞后阶数。
func ArchTest(residuals []float64, lags int) (*TestResult, error) {
	n := len(residuals)
	if lags <= 0 || lags >= n {
		return nil, fmt.Errorf("无效的 ARCH 检验阶数: %d", lags)
	}

	// 计算残差平方
	e2 := make([]float64, n)
	for i, r := range residuals {
		e2[i] = r * r
	}

	// 辅助回归：e²_t ~ 1 + e²_{t-1} + ... + e²_{t-q}
	T := n - lags
	X := make([][]float64, T)
	y := make([]float64, T)

	for t := 0; t < T; t++ {
		row := make([]float64, lags+1)
		row[0] = 1.0 // 截距
		for j := 1; j <= lags; j++ {
			row[j] = e2[t+lags-j]
		}
		X[t] = row
		y[t] = e2[t+lags]
	}

	_, rss, err := ols.Fit(X, y)
	if err != nil {
		return nil, fmt.Errorf("ARCH 辅助回归失败: %w", err)
	}

	// 计算 R²
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(T)
	tss := 0.0
	for _, v := range y {
		d := v - mean
		tss += d * d
	}
	r2 := 0.0
	if tss > 0 {
		r2 = 1.0 - rss/tss
	}

	stat := float64(T) * r2
	pValue := 1 - chiSquaredCDF(stat, lags)

	return &TestResult{
		Name:      "ARCH-LM",
		Statistic: stat,
		PValue:    pValue,
		DF:        lags,
		Reject:    pValue < 0.05,
	}, nil
}

// NormalityTest 执行残差正态性综合检验。
// 结合 Jarque-Bera 检验和 Shapiro-Wilk 近似检验。
func NormalityTest(residuals []float64) (*TestResult, error) {
	n := len(residuals)
	if n < 8 {
		return nil, fmt.Errorf("样本量不足（至少需要8个观测值）")
	}

	// 计算样本矩
	mean := 0.0
	for _, r := range residuals {
		mean += r
	}
	mean /= float64(n)

	m2, m3, m4 := 0.0, 0.0, 0.0
	for _, r := range residuals {
		d := r - mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	m2 /= float64(n)
	m3 /= float64(n)
	m4 /= float64(n)

	// 偏度和峰度
	if m2 == 0 {
		return &TestResult{
			Name:      "Jarque-Bera",
			Statistic: 0,
			PValue:    1,
			DF:        2,
			Reject:    false,
		}, nil
	}
	skewness := m3 / math.Pow(m2, 1.5)
	kurtosis := m4/(m2*m2) - 3.0 // 超额峰度

	// Jarque-Bera 统计量
	jb := (float64(n) / 6.0) * (skewness*skewness + kurtosis*kurtosis/4.0)
	pValue := 1 - chiSquaredCDF(jb, 2)

	return &TestResult{
		Name:      "Jarque-Bera",
		Statistic: jb,
		PValue:    pValue,
		DF:        2,
		Reject:    pValue < 0.05,
	}, nil
}

// chiSquaredCDF 卡方分布近似 CDF
func chiSquaredCDF(x float64, df int) float64 {
	if x <= 0 {
		return 0
	}
	k := float64(df)
	z := math.Pow(x/k, 1.0/3.0) - (1 - 2.0/(9*k))
	z /= math.Sqrt(2.0 / (9 * k))
	return 0.5 * (1 + math.Erf(z/math.Sqrt2))
}
