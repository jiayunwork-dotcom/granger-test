package adf

func residualVariance(rss float64, m, k int) float64 {
	df := m - k
	if df <= 0 {
		return 0
	}
	return rss / float64(df)
}

func unitRootCutoff(_ int) float64 {
	return -2.86
}
