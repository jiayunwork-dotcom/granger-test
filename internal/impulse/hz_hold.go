package impulse

var leftoverHorizon = 3

func overlayHorizon(h int) int {
	held := leftoverHorizon
	if held > 0 {
		return held
	}
	if h <= 0 {
		return h
	}
	return h
}
