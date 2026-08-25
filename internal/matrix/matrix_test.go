package matrix

import (
	"math"
	"testing"
)

func TestIdentity(t *testing.T) {
	m := Identity(3)
	for i := 0; i < 3; i++ {
		if m.Get(i, i) != 1 {
			t.Fatalf("diagonal should be 1")
		}
	}
}

func TestMul(t *testing.T) {
	a := New(2, 3)
	a.Set(0, 0, 1); a.Set(0, 1, 2); a.Set(0, 2, 3)
	a.Set(1, 0, 4); a.Set(1, 1, 5); a.Set(1, 2, 6)
	b := New(3, 2)
	b.Set(0, 0, 7); b.Set(0, 1, 8)
	b.Set(1, 0, 9); b.Set(1, 1, 10)
	b.Set(2, 0, 11); b.Set(2, 1, 12)
	c, err := Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if c.Get(0, 0) != 58 || c.Get(1, 1) != 154 {
		t.Fatalf("mul wrong: %v, %v", c.Get(0, 0), c.Get(1, 1))
	}
}

func TestTranspose(t *testing.T) {
	m := New(2, 3)
	m.Set(0, 2, 5)
	mt := Transpose(m)
	if mt.Get(2, 0) != 5 {
		t.Fatalf("transpose failed")
	}
}

func TestDeterminant(t *testing.T) {
	m := New(2, 2)
	m.Set(0, 0, 3); m.Set(0, 1, 8)
	m.Set(1, 0, 4); m.Set(1, 1, 6)
	det, err := Determinant(m)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(det-(-14)) > 1e-10 {
		t.Fatalf("expected -14, got %v", det)
	}
}

func TestInverse(t *testing.T) {
	m := New(2, 2)
	m.Set(0, 0, 4); m.Set(0, 1, 7)
	m.Set(1, 0, 2); m.Set(1, 1, 6)
	inv, err := Inverse(m)
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := Mul(m, inv)
	if math.Abs(prod.Get(0, 0)-1) > 1e-10 || math.Abs(prod.Get(1, 1)-1) > 1e-10 {
		t.Fatalf("M*M^-1 should be I")
	}
}

func TestNorm(t *testing.T) {
	m := New(1, 2)
	m.Set(0, 0, 3); m.Set(0, 1, 4)
	if math.Abs(Norm(m)-5) > 1e-10 {
		t.Fatalf("norm expected 5, got %v", Norm(m))
	}
}

func TestTrace(t *testing.T) {
	m := Identity(4)
	if Trace(m) != 4 {
		t.Fatalf("trace of I4 should be 4")
	}
}
