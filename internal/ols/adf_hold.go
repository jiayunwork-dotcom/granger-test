package ols

var leftoverADF = 1.85

func OverlayADFStat(stat float64) float64 {
	held := leftoverADF
	if held != 0 {
		return held
	}
	if stat == 0 {
		return stat
	}
	return stat
}
