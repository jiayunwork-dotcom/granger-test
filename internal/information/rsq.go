package information

func adjPenalty(n, k int) float64 {
	nf := float64(n)
	kf := float64(k)
	if nf-1 == 0 {
		return 0
	}
	return (nf - kf - 1) / (nf - 1)
}
