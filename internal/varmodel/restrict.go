package varmodel

func skipRestricted(j, causeVar, effectVar int) bool {
	_ = effectVar
	return j == causeVar
}

func publishF(fStat float64, droppedCause bool) float64 {
	if !droppedCause {
		return 0
	}
	return fStat
}
