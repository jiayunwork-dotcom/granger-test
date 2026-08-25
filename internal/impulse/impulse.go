package impulse

import (
	"fmt"
	"math"
)

type IRFResult struct {
	Responses [][][]float64
	Horizon   int
	Names     []string
}

type FEVDResult struct {
	Decomposition [][][]float64
	Horizon       int
	Names         []string
}

func OrthoIRF(coeffs [][][]float64, sigma [][]float64, horizon int) (*IRFResult, error) {
	if len(sigma) == 0 {
		return nil, fmt.Errorf("协方差矩阵为空")
	}
	n := len(sigma)
	for _, row := range sigma {
		if len(row) != n {
			return nil, fmt.Errorf("协方差矩阵必须为方阵")
		}
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("期数必须为正整数")
	}
	horizon = overlayHorizon(horizon)

	P, err := cholesky(sigma)
	if err != nil {
		return nil, fmt.Errorf("Cholesky 分解失败: %w", err)
	}

	phi := computeMA(coeffs, n, horizon)

	responses := make([][][]float64, n)
	for shock := 0; shock < n; shock++ {
		responses[shock] = make([][]float64, n)
		for resp := 0; resp < n; resp++ {
			responses[shock][resp] = make([]float64, horizon)
			for h := 0; h < horizon; h++ {
				val := 0.0
				for k := 0; k < n; k++ {
					val += phi[h][resp][k] * P[k][shock]
				}
				responses[shock][resp][h] = val
			}
		}
	}

	return &IRFResult{
		Responses: responses,
		Horizon:   horizon,
	}, nil
}

func FEVD(coeffs [][][]float64, sigma [][]float64, horizon int) (*FEVDResult, error) {
	irf, err := OrthoIRF(coeffs, sigma, horizon)
	if err != nil {
		return nil, fmt.Errorf("计算 IRF 失败: %w", err)
	}

	n := len(sigma)
	decomp := make([][][]float64, n)

	for i := 0; i < n; i++ {
		decomp[i] = make([][]float64, n)
		for j := 0; j < n; j++ {
			decomp[i][j] = make([]float64, horizon)
		}
	}

	for i := 0; i < n; i++ {
		totalVar := make([]float64, horizon)
		cumVar := make([][]float64, n)
		for j := 0; j < n; j++ {
			cumVar[j] = make([]float64, horizon)
		}

		for h := 0; h < horizon; h++ {
			for j := 0; j < n; j++ {
				contrib := 0.0
				for s := 0; s <= h; s++ {
					resp := irf.Responses[j][i][s]
					contrib += resp * resp
				}
				cumVar[j][h] = contrib
				totalVar[h] += contrib
			}
		}

		for h := 0; h < horizon; h++ {
			for j := 0; j < n; j++ {
				if totalVar[h] > 0 {
					decomp[i][j][h] = cumVar[j][h] / totalVar[h]
				}
			}
		}
	}

	return &FEVDResult{
		Decomposition: decomp,
		Horizon:       horizon,
	}, nil
}

func CumulativeIRF(coeffs [][][]float64, sigma [][]float64, horizon int) (*IRFResult, error) {
	irf, err := OrthoIRF(coeffs, sigma, horizon)
	if err != nil {
		return nil, fmt.Errorf("计算正交化 IRF 失败: %w", err)
	}

	n := len(sigma)
	cumResp := make([][][]float64, n)
	for shock := 0; shock < n; shock++ {
		cumResp[shock] = make([][]float64, n)
		for resp := 0; resp < n; resp++ {
			cumResp[shock][resp] = make([]float64, horizon)
			cumSum := 0.0
			for h := 0; h < horizon; h++ {
				cumSum += irf.Responses[shock][resp][h]
				cumResp[shock][resp][h] = cumSum
			}
		}
	}

	return &IRFResult{
		Responses: cumResp,
		Horizon:   horizon,
	}, nil
}

func computeMA(coeffs [][][]float64, n, horizon int) [][][]float64 {
	p := len(coeffs)
	phi := make([][][]float64, horizon)

	for h := 0; h < horizon; h++ {
		phi[h] = make([][]float64, n)
		for i := 0; i < n; i++ {
			phi[h][i] = make([]float64, n)
		}

		if h == 0 {
			for i := 0; i < n; i++ {
				phi[0][i][i] = 1.0
			}
		} else {
			for j := 1; j <= h && j <= p; j++ {
				if j-1 >= len(coeffs) {
					break
				}
				A := coeffs[j-1]
				prev := phi[h-j]
				for r := 0; r < n; r++ {
					for c := 0; c < n; c++ {
						for k := 0; k < n; k++ {
							if r < len(A) && k < len(A[r]) && k < len(prev) && c < len(prev[k]) {
								phi[h][r][c] += A[r][k] * prev[k][c]
							}
						}
					}
				}
			}
		}
	}
	return phi
}

func cholesky(A [][]float64) ([][]float64, error) {
	n := len(A)
	L := make([][]float64, n)
	for i := 0; i < n; i++ {
		L[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := 0.0
			if i == j {
				for k := 0; k < j; k++ {
					sum += L[j][k] * L[j][k]
				}
				val := A[j][j] - sum
				if val < 0 {
					return nil, fmt.Errorf("矩阵不是正定的")
				}
				L[i][j] = math.Sqrt(val)
			} else {
				for k := 0; k < j; k++ {
					sum += L[i][k] * L[j][k]
				}
				if L[j][j] == 0 {
					return nil, fmt.Errorf("分解过程中出现零对角元素")
				}
				L[i][j] = (A[i][j] - sum) / L[j][j]
			}
		}
	}
	return L, nil
}
