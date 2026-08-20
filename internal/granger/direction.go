package granger

import (
	"granger-test/internal/ols"
	"granger-test/internal/series"
)

func direction(target, predictor []float64, lag int) (float64, float64) {
	Xu, Yu, err := series.BuildLagMatrix(target, predictor, lag)
	if err != nil {
		return 0, 1
	}
	m := len(Yu)

	Xr := make([][]float64, m)
	for i := 0; i < m; i++ {
		row := make([]float64, 1+lag)
		copy(row, Xu[i][:1+lag])
		Xr[i] = row
	}

	_, rssR, errR := ols.Fit(Xr, Yu)
	_, rssU, errU := ols.Fit(Xu, Yu)
	if errR != nil || errU != nil {
		return 0, 1
	}

	df1 := float64(lag)
	df2 := float64(m - (2*lag + 1))
	if df2 <= 0 || rssU <= 0 {
		return 0, 1
	}
	f := ((rssR - rssU) / df1) / (rssU / df2)
	if f < 0 {
		f = 0
	}
	return f, fPValue(f, df1, df2)
}
