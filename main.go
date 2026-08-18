package main

import (
	"flag"
	"fmt"
	"os"

	"granger-test/internal/granger"
	"granger-test/internal/series"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	xPath := flag.String("x", "", "path to the X series CSV file")
	yPath := flag.String("y", "", "path to the Y series CSV file")
	lag := flag.Int("lag", 0, "lag order p (positive integer)")
	flag.Parse()

	if *xPath == "" || *yPath == "" {
		return fmt.Errorf("both -x and -y CSV paths are required")
	}
	if *lag <= 0 {
		return fmt.Errorf("lag must be a positive integer")
	}

	x, y, err := series.LoadPair(*xPath, *yPath)
	if err != nil {
		return fmt.Errorf("load series: %w", err)
	}

	res, err := granger.Test(x, y, *lag)
	if err != nil {
		return fmt.Errorf("granger test: %w", err)
	}

	fmt.Printf("Granger causality test (lag=%d)\n", *lag)
	fmt.Printf("  X->Y: F=%.4f p=%.4f\n", res.FX, res.PX)
	fmt.Printf("  Y->X: F=%.4f p=%.4f\n", res.FY, res.PY)
	fmt.Printf("  Direction: %s\n", res.Direction)
	return nil
}
