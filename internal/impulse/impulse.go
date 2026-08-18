// Package impulse 提供脉冲响应分析功能。
// 包括正交化脉冲响应函数（IRF）、预测误差方差分解（FEVD）和累积脉冲响应。
package impulse

import (
	"fmt"
	"math"
)

// IRFResult 保存脉冲响应函数计算结果
type IRFResult struct {
	// Responses[i][j][h] 表示第 j 个变量对第 i 个变量脉冲的第 h 期响应
	Responses [][][]float64
	// Horizon 为脉冲响应的计算期数
	Horizon int
	// Names 为变量名称
	Names []string
}

// FEVDResult 保存预测误差方差分解结果
type FEVDResult struct {
	// Decomposition[i][j][h] 表示第 i 个变量在第 h 期的预测误差中由第 j 个冲击贡献的比例
	Decomposition [][][]float64
	Horizon       int
	Names         []string
}

// OrthoIRF 计算正交化脉冲响应函数。
// coeffs 为 VAR 模型系数矩阵列表（每个滞后阶对应一个 n×n 矩阵），
// sigma 为残差协方差矩阵（n×n），
// horizon 为计算的响应期数。
// 使用 Cholesky 分解进行正交化。
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

	// Cholesky 分解: sigma = P * P'
	P, err := cholesky(sigma)
	if err != nil {
		return nil, fmt.Errorf("Cholesky 分解失败: %w", err)
	}

	// 计算移动平均表示的系数矩阵 Phi
	phi := computeMA(coeffs, n, horizon)

	// 正交化脉冲响应: Theta_h = Phi_h * P
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

// FEVD 计算预测误差方差分解。
// 基于正交化脉冲响应函数，分解各变量预测误差的来源。
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

	// 对每个变量 i 的预测误差方差进行分解
	for i := 0; i < n; i++ {
		// 累计总方差
		totalVar := make([]float64, horizon)
		cumVar := make([][]float64, n)
		for j := 0; j < n; j++ {
			cumVar[j] = make([]float64, horizon)
		}

		for h := 0; h < horizon; h++ {
			for j := 0; j < n; j++ {
				// 第 j 个冲击到第 i 个变量在 0..h 期的累计贡献
				contrib := 0.0
				for s := 0; s <= h; s++ {
					resp := irf.Responses[j][i][s]
					contrib += resp * resp
				}
				cumVar[j][h] = contrib
				totalVar[h] += contrib
			}
		}

		// 归一化为比例
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

// CumulativeIRF 计算累积脉冲响应函数。
// 累积 IRF 是各期正交化 IRF 的逐步累加，用于分析长期效应。
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

// computeMA 计算 VAR 模型的移动平均（MA）表示系数矩阵。
// 返回 phi[h] 为第 h 期的 n×n 系数矩阵。
func computeMA(coeffs [][][]float64, n, horizon int) [][][]float64 {
	p := len(coeffs) // VAR 阶数
	phi := make([][][]float64, horizon)

	for h := 0; h < horizon; h++ {
		phi[h] = make([][]float64, n)
		for i := 0; i < n; i++ {
			phi[h][i] = make([]float64, n)
		}

		if h == 0 {
			// Phi_0 = I（单位矩阵）
			for i := 0; i < n; i++ {
				phi[0][i][i] = 1.0
			}
		} else {
			// Phi_h = sum_{j=1}^{min(h,p)} A_j * Phi_{h-j}
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

// cholesky 执行 Cholesky 分解，返回下三角矩阵 L 使得 A = L * L'
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
