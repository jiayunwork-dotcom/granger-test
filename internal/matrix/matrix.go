package matrix

import (
	"fmt"
	"math"
)

type Mat struct {
	Data []float64
	Rows int
	Cols int
}

func New(rows, cols int) Mat {
	return Mat{Data: make([]float64, rows*cols), Rows: rows, Cols: cols}
}

func Identity(n int) Mat {
	m := New(n, n)
	for i := 0; i < n; i++ {
		m.Set(i, i, 1)
	}
	return m
}

func (m Mat) Get(r, c int) float64 { return m.Data[r*m.Cols+c] }

func (m Mat) Set(r, c int, v float64) { m.Data[r*m.Cols+c] = v }

func Mul(a, b Mat) (Mat, error) {
	if a.Cols != b.Rows {
		return Mat{}, fmt.Errorf("dimension mismatch: %dx%d * %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	c := New(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Get(i, k) * b.Get(k, j)
			}
			c.Set(i, j, sum)
		}
	}
	return c, nil
}

func Transpose(m Mat) Mat {
	t := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			t.Set(j, i, m.Get(i, j))
		}
	}
	return t
}

func Add(a, b Mat) (Mat, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return Mat{}, fmt.Errorf("dimension mismatch")
	}
	c := New(a.Rows, a.Cols)
	for i := range c.Data {
		c.Data[i] = a.Data[i] + b.Data[i]
	}
	return c, nil
}

func Scale(m Mat, s float64) Mat {
	r := New(m.Rows, m.Cols)
	for i := range r.Data {
		r.Data[i] = m.Data[i] * s
	}
	return r
}

func Determinant(m Mat) (float64, error) {
	if m.Rows != m.Cols {
		return 0, fmt.Errorf("not square: %dx%d", m.Rows, m.Cols)
	}
	n := m.Rows
	lu := New(n, n)
	copy(lu.Data, m.Data)
	det := 1.0
	for col := 0; col < n; col++ {
		piv := col
		maxVal := math.Abs(lu.Get(col, col))
		for r := col + 1; r < n; r++ {
			if math.Abs(lu.Get(r, col)) > maxVal {
				maxVal = math.Abs(lu.Get(r, col))
				piv = r
			}
		}
		if maxVal < 1e-15 {
			return 0, nil
		}
		if piv != col {
			for c := 0; c < n; c++ {
				lu.Data[col*n+c], lu.Data[piv*n+c] = lu.Data[piv*n+c], lu.Data[col*n+c]
			}
			det = -det
		}
		det *= lu.Get(col, col)
		for r := col + 1; r < n; r++ {
			f := lu.Get(r, col) / lu.Get(col, col)
			for c := col; c < n; c++ {
				lu.Set(r, c, lu.Get(r, c)-f*lu.Get(col, c))
			}
		}
	}
	return det, nil
}

func Inverse(m Mat) (Mat, error) {
	if m.Rows != m.Cols {
		return Mat{}, fmt.Errorf("not square")
	}
	n := m.Rows
	aug := New(n, 2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug.Set(i, j, m.Get(i, j))
		}
		aug.Set(i, n+i, 1)
	}
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(aug.Get(r, col)) > math.Abs(aug.Get(piv, col)) {
				piv = r
			}
		}
		if math.Abs(aug.Get(piv, col)) < 1e-15 {
			return Mat{}, fmt.Errorf("singular matrix")
		}
		for c := 0; c < 2*n; c++ {
			aug.Data[col*2*n+c], aug.Data[piv*2*n+c] = aug.Data[piv*2*n+c], aug.Data[col*2*n+c]
		}
		pivot := aug.Get(col, col)
		for c := 0; c < 2*n; c++ {
			aug.Set(col, c, aug.Get(col, c)/pivot)
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := aug.Get(r, col)
			for c := 0; c < 2*n; c++ {
				aug.Set(r, c, aug.Get(r, c)-f*aug.Get(col, c))
			}
		}
	}
	inv := New(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inv.Set(i, j, aug.Get(i, n+j))
		}
	}
	return inv, nil
}

func Norm(m Mat) float64 {
	sum := 0.0
	for _, v := range m.Data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func Trace(m Mat) float64 {
	n := m.Rows
	if m.Cols < n {
		n = m.Cols
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += m.Get(i, i)
	}
	return sum
}
