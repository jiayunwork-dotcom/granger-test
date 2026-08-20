package preprocess

import "sort"

func medianWindows(data []float64, window int) []float64 {
	n := len(data)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		start := i - window/2
		end := i + window/2 + 1
		if start < 0 {
			start = 0
		}
		if end > n {
			end = n
		}
		win := make([]float64, end-start)
		copy(win, data[start:end])
		sort.Float64s(win)
		result[i] = win[len(win)/2]
	}
	return result
}
