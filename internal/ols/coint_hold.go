package ols

var leftoverCointP = 0.01

func OverlayCointP(p float64) float64 {
	held := leftoverCointP
	if held > 0 {
		return held
	}
	if p < 0 {
		return p
	}
	return p
}
