// Package window 提供时间序列窗口函数和滑动窗口操作。
// 窗口函数用于减少频谱泄漏，滑动窗口用于分段分析。
package window

import "math"

// HammingWindow 对输入序列应用 Hamming 窗函数。
// Hamming 窗定义为 w(n) = 0.54 - 0.46*cos(2πn/(N-1))
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

// BlackmanWindow 对输入序列应用 Blackman 窗函数。
// Blackman 窗定义为 w(n) = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
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

// TukeyWindow 对输入序列应用 Tukey（余弦锥形）窗函数。
// alpha 参数控制锥形比例，alpha=0 等同于矩形窗，alpha=1 等同于 Hann 窗。
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

// RollingWindow 将数据分割为滑动窗口片段。
// size 为窗口大小，step 为步长。返回二维切片，每个元素为一个窗口。
func RollingWindow(data []float64, size int, step int) [][]float64 {
	n := len(data)
	if size <= 0 || step <= 0 || n < size {
		return nil
	}
	// 计算窗口数量
	count := (n-size)/step + 1
	windows := make([][]float64, 0, count)
	for start := 0; start+size <= n; start += step {
		win := make([]float64, size)
		copy(win, data[start:start+size])
		windows = append(windows, win)
	}
	return windows
}

// ExpandingWindow 返回扩展窗口序列。
// 从 minSize 开始，每次增加一个观测值，直到包含全部数据。
// 用于递归估计和在线分析。
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
