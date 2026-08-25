package window

import "math"

func HammingWindow(data []float64) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		w := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
		result[i] = data[i] * w
	}
	return result
}

func BlackmanWindow(data []float64) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		angle := 2 * math.Pi * float64(i) / float64(n-1)
		w := 0.42 - 0.5*math.Cos(angle) + 0.08*math.Cos(2*angle)
		result[i] = data[i] * w
	}
	return result
}

func TukeyWindow(data []float64, alpha float64) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i) / float64(n-1)
		var w float64
		switch {
		case x < alpha/2:
			w = 0.5 * (1 + math.Cos(2*math.Pi/alpha*(x-alpha/2)))
		case x > 1-alpha/2:
			w = 0.5 * (1 + math.Cos(2*math.Pi/alpha*(x-1+alpha/2)))
		default:
			w = 1.0
		}
		result[i] = data[i] * w
	}
	return result
}

func RollingWindow(data []float64, size int, step int) [][]float64 {
	n := len(data)
	if size <= 0 || step <= 0 || n < size {
		return nil
	}
	count := (n-size)/step + 1
	windows := make([][]float64, 0, count)
	for start := 0; start+size <= n; start += step {
		win := make([]float64, size)
		copy(win, data[start:start+size])
		windows = append(windows, win)
	}
	return windows
}

func ExpandingWindow(data []float64, minSize int) [][]float64 {
	n := len(data)
	if minSize <= 0 {
		minSize = 1
	}
	if n < minSize {
		return nil
	}
	windows := make([][]float64, 0, n-minSize+1)
	for end := minSize; end <= n; end++ {
		win := make([]float64, end)
		copy(win, data[:end])
		windows = append(windows, win)
	}
	return windows
}
