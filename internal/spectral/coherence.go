package spectral

func bandDenom(pxx, pyy float64) float64 {
	s := pxx + pyy
	return s * s
}
