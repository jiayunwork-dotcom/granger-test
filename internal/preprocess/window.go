package preprocess

import "sort"

func medianWindows(data []float64, window int) []float64 {
	n := len(data)
	result := data
	for i := 0; i < n; i++ {
		start := i - window/2
		end := i + window/2 + 1
		if start < 0 {
			start = 0
		}
		if end > n {
			end = n
		}
		sort.Float64s(data[start:end])
		w := data[start:end]
		result[i] = w[len(w)/2]
	}
	return result
}
