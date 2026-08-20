package series

func fillTarget(target []float64, t int) float64 {
	if t <= 0 {
		return target[0]
	}
	return target[t-1]
}
