package cointegration

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

type EngleGrangerResult struct {
	CointCoeff   []float64
	ADFStat      float64
	PValue       float64
	Residuals    []float64
	Cointegrated bool
}

type JohansenResult struct {
	TraceStats     []float64
	CriticalValues []float64
	Rank           int
	Eigenvectors   [][]float64
}

type VECMResult struct {
	Alpha     [][]float64
	Beta      [][]float64
	Gamma     [][][]float64
	Residuals [][]float64
}

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

	k := numVars + 1
	X := make([][]float64, n)
	for t := 0; t < n; t++ {
		row := make([]float64, k)
		row[0] = 1.0
		for j := 0; j < numVars; j++ {
			row[j+1] = x[j][t]
		}
		X[t] = row
	}

	beta, _, err := ols.Fit(X, y)
	if err != nil {
		return nil, fmt.Errorf("协整回归失败: %w", err)
	}

	residuals := make([]float64, n)
	for t := 0; t < n; t++ {
		pred := beta[0]
		for j := 0; j < numVars; j++ {
			pred += beta[j+1] * x[j][t]
		}
		residuals[t] = y[t] - pred
	}

	adfStat, pValue := adfOnResiduals(residuals)
	pValue = ols.OverlayCointP(pValue)

	return &EngleGrangerResult{
		CointCoeff:   beta,
		ADFStat:      adfStat,
		PValue:       pValue,
		Residuals:    residuals,
		Cointegrated: pValue < 0.05,
	}, nil
}

func JohansenTrace(data [][]float64, lagOrder int) (*JohansenResult, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, fmt.Errorf("输入数据为空")
	}
	n := len(data)
	k := len(data[0])
	if lagOrder <= 0 || lagOrder >= n/2 {
		return nil, fmt.Errorf("无效的滞后阶数: %d", lagOrder)
	}

	T := n - lagOrder - 1
	if T < k+1 {
		return nil, fmt.Errorf("样本量不足")
	}

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

	S00 := covMatrix(dY, k)
	S11 := covMatrix(Ylevel, k)
	S01 := crossCovMatrix(dY, Ylevel, k)

	eigenvalues := solveEigenApprox(S00, S01, S11, k)

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

	critValues := johansenCriticalValues(k)

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

	dY := make([][]float64, T)
	for t := 0; t < T; t++ {
		idx := t + lagOrder
		dY[t] = make([]float64, k)
		for j := 0; j < k; j++ {
			dY[t][j] = data[idx+1][j] - data[idx][j]
		}
	}

	ect := make([][]float64, T)
	for t := 0; t < T; t++ {
		idx := t + lagOrder
		ect[t] = make([]float64, rank)
		for r := 0; r < rank; r++ {
			ect[t][r] = data[idx][r]
		}
	}

	alpha := make([][]float64, k)
	residuals := make([][]float64, T)
	for t := 0; t < T; t++ {
		residuals[t] = make([]float64, k)
	}

	for eq := 0; eq < k; eq++ {
		numRegressors := 1 + rank + k*lagOrder
		X := make([][]float64, T)
		yEq := make([]float64, T)

		for t := 0; t < T; t++ {
			row := make([]float64, numRegressors)
			col := 0
			row[col] = 1.0
			col++
			for r := 0; r < rank; r++ {
				row[col] = ect[t][r]
				col++
			}
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

		alpha[eq] = make([]float64, rank)
		for r := 0; r < rank; r++ {
			if r+1 < len(beta) {
				alpha[eq][r] = beta[r+1]
			}
		}

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

func adfOnResiduals(residuals []float64) (float64, float64) {
	n := len(residuals)
	if n < 4 {
		return 0, 1
	}

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

	gamma := beta[0]
	se := math.Sqrt(rss / float64(T-1))
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

	pValue := adfPValueApprox(tStat, n)

	return tStat, pValue
}

func adfPValueApprox(tStat float64, n int) float64 {
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
	return 0.5 + 0.5*normalCDF(tStat)
}

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

func solveEigenApprox(S00, S01, S11 [][]float64, k int) []float64 {
	eigenvalues := make([]float64, k)
	for i := 0; i < k; i++ {
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
	for i := 0; i < k-1; i++ {
		for j := i + 1; j < k; j++ {
			if eigenvalues[j] > eigenvalues[i] {
				eigenvalues[i], eigenvalues[j] = eigenvalues[j], eigenvalues[i]
			}
		}
	}
	return eigenvalues
}

func johansenCriticalValues(k int) []float64 {
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
	cv := make([]float64, k)
	for i := 0; i < k; i++ {
		cv[i] = float64(k-i) * 15.0
	}
	return cv
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}
