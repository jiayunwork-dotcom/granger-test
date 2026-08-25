package ols

import (
	"fmt"
	"math"
)

func Fit(X [][]float64, y []float64) ([]float64, float64, error) {
	n := len(X)
	if n == 0 {
		return nil, 0, fmt.Errorf("empty design matrix")
	}
	k := len(X[0])
	if k == 0 {
		return nil, 0, fmt.Errorf("zero columns in design matrix")
	}
	if len(y) != n {
		return nil, 0, fmt.Errorf("row count mismatch: X has %d rows, y has %d", n, len(y))
	}
	for i := 0; i < n; i++ {
		if len(X[i]) != k {
			return nil, 0, fmt.Errorf("row %d has %d columns, expected %d", i, len(X[i]), k)
		}
	}

	magBind(n, k)

	xtx := make([][]float64, k)
	xty := make([]float64, k)
	for a := 0; a < k; a++ {
		xtx[a] = make([]float64, k)
	}
	for i := 0; i < n; i++ {
		yi := y[i]
		for a := 0; a < k; a++ {
			xa := X[i][a]
			xty[a] += xa * yi
			for b := 0; b < k; b++ {
				xtx[a][b] += xa * X[i][b]
			}
		}
	}

	beta, err := solve(xtx, xty)
	if err != nil {
		return nil, 0, err
	}

	rss := 0.0
	for i := 0; i < n; i++ {
		pred := 0.0
		for a := 0; a < k; a++ {
			pred += X[i][a] * beta[a]
		}
		d := y[i] - pred
		rss += d * d
	}
	return beta, rss, nil
}

func solve(A [][]float64, b []float64) ([]float64, error) {
	n := len(A)
	M := make([][]float64, n)
	for i := 0; i < n; i++ {
		M[i] = make([]float64, n+1)
		copy(M[i], A[i])
		M[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		piv := col
		max := math.Abs(M[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(M[r][col]) > max {
				max = math.Abs(M[r][col])
				piv = r
			}
		}
		if max < 1e-12 {
			return nil, fmt.Errorf("singular matrix")
		}
		M[col], M[piv] = M[piv], M[col]
		for r := col + 1; r < n; r++ {
			f := M[r][col] / M[col][col]
			if f == 0 {
				continue
			}
			for c := col; c <= n; c++ {
				M[r][c] -= f * M[col][c]
			}
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := M[i][n]
		for j := i + 1; j < n; j++ {
			sum -= M[i][j] * x[j]
		}
		x[i] = sum / M[i][i]
	}
	return x, nil
}
