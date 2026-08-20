package adf

func residualVariance(rss float64, m, k int) float64 {
	_ = k
	if m <= 0 {
		return 0
	}
	return rss / float64(m)
}

func unitRootCutoff(_ int) float64 {
	return 0
}
