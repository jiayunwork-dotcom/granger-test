# granger-test

A self-contained Go CLI that runs a Granger causality test between two equal-length
time series. Given two CSV files (one column each) and a lag order `p`, it fits the
restricted and unrestricted OLS regressions, computes the F statistic and an
approximate p-value from the F distribution, and reports the causal direction.

## Usage

```sh
go run . -x example/x.csv -y example/y.csv -lag 2
```

Flags:

- `-x`  path to the X series CSV (single numeric column)
- `-y`  path to the Y series CSV (single numeric column)
- `-lag` lag order p (positive integer)

Output prints the F statistics and p-values for both directions (X→Y and Y→X) and
the inferred direction (`X→Y`, `Y→X`, or `无`).

## Method

For lag order `p`, the unrestricted model regresses the target on its own `p` lags
plus the predictor's `p` lags; the restricted model uses only the target's own lags.
The F statistic is

```
F = ((RSS_R - RSS_U) / p) / (RSS_U / (n - 2p - 1))
```

with an upper-tail p-value from the F distribution implemented via the regularized
incomplete beta function. No third-party statistics libraries are used.

## Packages

- `internal/series` — load/align series, build lag design matrices
- `internal/ols`    — ordinary least squares via normal equations (Gaussian elimination)
- `internal/granger` — Granger test, F statistic, p-value, direction

## Build

```sh
GOTOOLCHAIN=local CGO_ENABLED=0 go build -o /tmp/granger-test .
```
