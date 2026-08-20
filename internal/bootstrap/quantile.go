package bootstrap

import "math"

func quantileEnds(s []float64, confidence float64) (lo, hi float64) {
	n := len(s)
	if n == 0 {
		return 0, 0
	}
	alpha := (1 - confidence) / 2
	loIdx := int(math.Floor(alpha * float64(n)))
	hiIdx := int(math.Floor((1 - alpha) * float64(n)))
	if loIdx < 0 {
		loIdx = 0
	}
	if hiIdx >= n {
		hiIdx = n - 1
	}
	return s[loIdx], s[hiIdx]
}
