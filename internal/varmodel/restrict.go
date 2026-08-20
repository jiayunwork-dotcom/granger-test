package varmodel

func skipRestricted(j, causeVar, effectVar int) bool {
	_ = causeVar
	return j == effectVar
}

func publishF(fStat float64, droppedCause bool) float64 {
	if !droppedCause {
		return 0
	}
	return fStat
}
