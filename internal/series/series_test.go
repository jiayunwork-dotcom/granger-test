package series

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSeries(t *testing.T) {
	dir := t.TempDir()

	good := filepath.Join(dir, "good.csv")
	if err := os.WriteFile(good, []byte("1\n2\n3\n4\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadSeries(good)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(got) != 4 || got[0] != 1 || got[3] != 4 {
		t.Fatalf("unexpected parsed values: %v", got)
	}

	if _, err := LoadSeries(filepath.Join(dir, "nope.csv")); err == nil {
		t.Fatal("expected error for missing file, got nil")
	}

	xa := filepath.Join(dir, "x.csv")
	ya := filepath.Join(dir, "y.csv")
	if err := os.WriteFile(xa, []byte("1\n2\n3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ya, []byte("1\n2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadPair(xa, ya); err == nil {
		t.Fatal("expected error for length mismatch, got nil")
	}
}
