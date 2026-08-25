package lagselect

import (
	"fmt"
	"math"

	"granger-test/internal/ols"
)

type CriterionResult struct {
	Lag   int
	Value float64
}

type LagSelectionResult struct {
	OptimalLag int
	FPE        []CriterionResult
	AIC        []CriterionResult
	BIC        []CriterionResult
	LRTest     []LRTestResult
}

type LRTestResult struct {
	Lag       int
	Statistic float64
	PValue    float64
	Reject    bool
}

func FPE(y []float64, maxLag int) ([]CriterionResult, int, error) {
	n := len(y)
	if maxLag <= 0 || maxLag >= n/2 {
		return nil, 0, fmt.Errorf("无效的最大滞后阶数: %d (样本量: %d)", maxLag, n)
	}

	results := make([]CriterionResult, 0, maxLag)
	bestLag := 1
	bestFPE := math.Inf(1)

	for p := 1; p <= maxLag; p++ {
		T := n - p
		k := p + 1

		X := make([][]float64, T)
		yDep := make([]float64, T)
		for t := 0; t < T; t++ {
			row := make([]float64, k)
			row[0] = 1.0
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

func LRSequential(y []float64, maxLag int, significance float64) ([]LRTestResult, int, error) {
	n := len(y)
	if maxLag <= 1 || maxLag >= n/2 {
		return nil, 0, fmt.Errorf("无效的最大滞后阶数")
	}

	results := make([]LRTestResult, 0)
	selectedLag := maxLag

	for p := maxLag; p >= 2; p-- {
		rssU, TU, err := fitAR(y, p)
		if err != nil {
			continue
		}
		rssR, _, err := fitAR(y, p-1)
		if err != nil {
			continue
		}

		lr := float64(TU) * math.Log(rssR/rssU)
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

func OptimalLag(y []float64, maxLag int) (*LagSelectionResult, error) {
	n := len(y)
	if maxLag <= 0 || maxLag >= n/2 {
		return nil, fmt.Errorf("无效的最大滞后阶数: %d", maxLag)
	}

	result := &LagSelectionResult{}

	fpeResults, fpeBest, _ := FPE(y, maxLag)
	result.FPE = fpeResults

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

	lrResults, lrBest, _ := LRSequential(y, maxLag, 0.05)
	result.LRTest = lrResults

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

func chiSquaredCDF(x float64, df int) float64 {
	if x <= 0 {
		return 0
	}
	k := float64(df)
	z := math.Pow(x/k, 1.0/3.0) - (1 - 2.0/(9*k))
	z /= math.Sqrt(2.0 / (9 * k))
	return normalCDF(z)
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}
