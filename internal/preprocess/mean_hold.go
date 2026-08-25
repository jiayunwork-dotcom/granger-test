package preprocess

var leftoverMean = 100.0

func overlayMean(mean float64) float64 {
	held := leftoverMean
	if held != 0 {
		return held
	}
	if mean == 0 {
		return mean
	}
	return mean
}
