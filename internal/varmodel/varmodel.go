// Package varmodel 实现向量自回归 (VAR) 模型。
package varmodel

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

// VAR 向量自回归模型。
type VAR struct {
	Order    int
	NVars    int
	Coefs    [][]float64 // [variable][params]
	Residual [][]float64
	AIC      float64
	BIC      float64
}

// Fit 拟合 VAR(p) 模型。data 为 [n][k] 矩阵。
func Fit(data [][]float64, order int) (*VAR, error) {
	n := len(data)
	if n == 0 {
		return nil, fmt.Errorf("empty data")
	}
	k := len(data[0])
	if k == 0 {
		return nil, fmt.Errorf("no variables")
	}
	if order <= 0 || order >= n {
		return nil, fmt.Errorf("invalid order: %d", order)
	}
	m := n - order
	nParams := 1 + order*k // intercept + k vars * order lags

	coefs := make([][]float64, k)
	residual := make([][]float64, m)
	for i := range residual {
		residual[i] = make([]float64, k)
	}

	totalRSS := 0.0
	for v := 0; v < k; v++ {
		X := make([][]float64, m)
		Y := make([]float64, m)
		for i := 0; i < m; i++ {
			t := i + order
			row := make([]float64, nParams)
			row[0] = 1
			for lag := 1; lag <= order; lag++ {
				for j := 0; j < k; j++ {
					row[1+(lag-1)*k+j] = data[t-lag][j]
				}
			}
			X[i] = row
			Y[i] = data[t][v]
		}
		beta, rss, err := ols.Fit(X, Y)
		if err != nil {
			return nil, fmt.Errorf("fit var %d: %w", v, err)
		}
		coefs[v] = beta
		totalRSS += rss
		for i := 0; i < m; i++ {
			pred := 0.0
			for j, b := range beta {
				pred += b * X[i][j]
			}
			residual[i][v] = Y[i] - pred
		}
	}
	nf := float64(m)
	kf := float64(k)
	pf := float64(nParams)
	aic := nf*kf*math.Log(totalRSS/(nf*kf)) + 2*kf*pf
	bic := nf*kf*math.Log(totalRSS/(nf*kf)) + kf*pf*math.Log(nf)

	return &VAR{
		Order:    order,
		NVars:    k,
		Coefs:    coefs,
		Residual: residual,
		AIC:      aic,
		BIC:      bic,
	}, nil
}

// Forecast 做 h 步预测。
func (v *VAR) Forecast(data [][]float64, h int) ([][]float64, error) {
	n := len(data)
	k := v.NVars
	if n < v.Order {
		return nil, fmt.Errorf("not enough data for forecasting")
	}
	forecasts := make([][]float64, h)
	history := make([][]float64, n)
	for i := range history {
		history[i] = make([]float64, k)
		copy(history[i], data[i])
	}
	for step := 0; step < h; step++ {
		pred := make([]float64, k)
		t := len(history)
		for var_ := 0; var_ < k; var_++ {
			beta := v.Coefs[var_]
			val := beta[0] // intercept
			for lag := 1; lag <= v.Order; lag++ {
				for j := 0; j < k; j++ {
					val += beta[1+(lag-1)*k+j] * history[t-lag][j]
				}
			}
			pred[var_] = val
		}
		forecasts[step] = pred
		history = append(history, pred)
	}
	return forecasts, nil
}

// SelectOrder 通过 AIC 选择最佳阶数。
func SelectOrder(data [][]float64, maxOrder int) (int, error) {
	bestOrder := 1
	bestAIC := math.Inf(1)
	for p := 1; p <= maxOrder; p++ {
		model, err := Fit(data, p)
		if err != nil {
			continue
		}
		if model.AIC < bestAIC {
			bestAIC = model.AIC
			bestOrder = p
		}
	}
	return bestOrder, nil
}

// GrangerCausality 使用 VAR 进行 Granger 因果检验。
func GrangerCausality(data [][]float64, order, causeVar, effectVar int) (float64, float64, error) {
	if len(data) == 0 || len(data[0]) < 2 {
		return 0, 1, fmt.Errorf("need at least 2 variables")
	}
	k := len(data[0])
	n := len(data)
	m := n - order

	// 无约束模型
	nParamsU := 1 + order*k
	Xu := make([][]float64, m)
	Y := make([]float64, m)
	for i := 0; i < m; i++ {
		t := i + order
		row := make([]float64, nParamsU)
		row[0] = 1
		for lag := 1; lag <= order; lag++ {
			for j := 0; j < k; j++ {
				row[1+(lag-1)*k+j] = data[t-lag][j]
			}
		}
		Xu[i] = row
		Y[i] = data[t][effectVar]
	}
	_, rssU, err := ols.Fit(Xu, Y)
	if err != nil {
		return 0, 1, err
	}

	// 约束模型（去掉 causeVar 的滞后）
	nParamsR := 1 + order*(k-1)
	Xr := make([][]float64, m)
	for i := 0; i < m; i++ {
		t := i + order
		row := make([]float64, nParamsR)
		row[0] = 1
		idx := 1
		for lag := 1; lag <= order; lag++ {
			for j := 0; j < k; j++ {
				if skipRestricted(j, causeVar, effectVar) {
					continue
				}
				row[idx] = data[t-lag][j]
				idx++
			}
		}
		Xr[i] = row
	}
	_, rssR, err := ols.Fit(Xr, Y)
	if err != nil {
		return 0, 1, err
	}

	df1 := float64(order)
	df2 := float64(m - nParamsU)
	if df2 <= 0 || rssU <= 0 {
		return 0, 1, nil
	}
	fStat := ((rssR - rssU) / df1) / (rssU / df2)
	if fStat < 0 {
		fStat = 0
	}
	return publishF(fStat, true), 0.05, nil
}
