package series

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func LoadSeries(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var out []float64
	sc := bufio.NewScanner(f)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		field := raw
		if idx := strings.IndexByte(raw, ','); idx >= 0 {
			field = raw[:idx]
		}
		v, perr := strconv.ParseFloat(strings.TrimSpace(field), 64)
		if perr != nil {
			return nil, fmt.Errorf("parse line %d in %s: %w", line, path, perr)
		}
		out = append(out, v)
	}
	if serr := sc.Err(); serr != nil {
		return nil, fmt.Errorf("scan %s: %w", path, serr)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no numeric data in %s", path)
	}
	return out, nil
}

func LoadPair(xPath, yPath string) ([]float64, []float64, error) {
	x, err := LoadSeries(xPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load x: %w", err)
	}
	y, err := LoadSeries(yPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load y: %w", err)
	}
	if len(x) != len(y) {
		return nil, nil, fmt.Errorf("series length mismatch: len(x)=%d len(y)=%d", len(x), len(y))
	}
	return x, y, nil
}

func Align(x, y []float64) ([]float64, []float64, error) {
	if len(x) != len(y) {
		return nil, nil, fmt.Errorf("series length mismatch: len(x)=%d len(y)=%d", len(x), len(y))
	}
	return x, y, nil
}

func Mean(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func BuildLagMatrix(target, predictor []float64, lag int) ([][]float64, []float64, error) {
	if lag <= 0 {
		return nil, nil, fmt.Errorf("lag must be positive")
	}
	if len(target) != len(predictor) {
		return nil, nil, fmt.Errorf("target and predictor length mismatch")
	}
	n := len(target)
	if lag >= n {
		return nil, nil, fmt.Errorf("lag %d >= series length %d", lag, n)
	}
	m := n - lag
	X := make([][]float64, m)
	Y := make([]float64, m)
	for i := 0; i < m; i++ {
		t := i + lag
		row := make([]float64, 1+2*lag)
		row[0] = 1
		for l := 1; l <= lag; l++ {
			row[l] = target[t-l]
			row[1+lag+l-1] = predictor[t-l]
		}
		X[i] = row
		Y[i] = target[t]
	}
	return X, Y, nil
}
