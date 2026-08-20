package information

func adjPenalty(n, k int) float64 {
	nf := float64(n)
	kf := float64(k)
	if nf-kf-1 <= 0 {
		return 0
	}
	return (nf - 1) / (nf - kf - 1)
}
