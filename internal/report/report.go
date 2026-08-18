// Package report 提供分析结果的格式化输出功能。
// 支持文本表格和 JSON 两种输出格式。
package report

import (
	"encoding/json"
	"fmt"
	"strings"
)

// GrangerResult 表示格兰杰因果检验的结果
type GrangerResult struct {
	Cause     string  `json:"cause"`
	Effect    string  `json:"effect"`
	Lag       int     `json:"lag"`
	FStatistic float64 `json:"f_statistic"`
	PValue    float64 `json:"p_value"`
	Reject    bool    `json:"reject"`
}

// ADFResult 表示 ADF 单位根检验的结果
type ADFResult struct {
	Variable   string  `json:"variable"`
	Statistic  float64 `json:"statistic"`
	PValue     float64 `json:"p_value"`
	Lags       int     `json:"lags"`
	Stationary bool    `json:"stationary"`
}

// VARSummary 表示 VAR 模型摘要信息
type VARSummary struct {
	Order      int       `json:"order"`
	NumVars    int       `json:"num_vars"`
	NumObs     int       `json:"num_obs"`
	AIC        float64   `json:"aic"`
	BIC        float64   `json:"bic"`
	Variables  []string  `json:"variables"`
	R2         []float64 `json:"r_squared"`
}

// FormatGranger 将格兰杰因果检验结果格式化为文本表格。
func FormatGranger(results []GrangerResult) string {
	if len(results) == 0 {
		return "无检验结果"
	}

	var sb strings.Builder
	sb.WriteString("╔══════════════════════════════════════════════════════════════════╗\n")
	sb.WriteString("║              格兰杰因果检验结果 (Granger Causality)              ║\n")
	sb.WriteString("╠══════════════╦══════════════╦═════╦══════════╦═════════╦════════╣\n")
	sb.WriteString("║ 原因变量     ║ 结果变量     ║ 滞后║ F统计量  ║ P值     ║ 结论   ║\n")
	sb.WriteString("╠══════════════╬══════════════╬═════╬══════════╬═════════╬════════╣\n")

	for _, r := range results {
		conclusion := "不拒绝"
		if r.Reject {
			conclusion = "拒绝H0"
		}
		sb.WriteString(fmt.Sprintf("║ %-12s ║ %-12s ║ %3d ║ %8.4f ║ %7.4f ║ %-6s ║\n",
			truncate(r.Cause, 12),
			truncate(r.Effect, 12),
			r.Lag,
			r.FStatistic,
			r.PValue,
			conclusion,
		))
	}

	sb.WriteString("╚══════════════╩══════════════╩═════╩══════════╩═════════╩════════╝\n")
	sb.WriteString(fmt.Sprintf("注: 显著性水平 α = 0.05，共 %d 组检验\n", len(results)))
	return sb.String()
}

// FormatADF 将 ADF 检验结果格式化为文本表格。
func FormatADF(results []ADFResult) string {
	if len(results) == 0 {
		return "无检验结果"
	}

	var sb strings.Builder
	sb.WriteString("┌─────────────────────────────────────────────────────┐\n")
	sb.WriteString("│          ADF 单位根检验结果 (Unit Root Test)         │\n")
	sb.WriteString("├──────────────┬──────────┬─────────┬─────┬──────────┤\n")
	sb.WriteString("│ 变量         │ 统计量   │ P值     │ 滞后│ 结论     │\n")
	sb.WriteString("├──────────────┼──────────┼─────────┼─────┼──────────┤\n")

	for _, r := range results {
		conclusion := "非平稳"
		if r.Stationary {
			conclusion = "平稳"
		}
		sb.WriteString(fmt.Sprintf("│ %-12s │ %8.4f │ %7.4f │ %3d │ %-8s │\n",
			truncate(r.Variable, 12),
			r.Statistic,
			r.PValue,
			r.Lags,
			conclusion,
		))
	}

	sb.WriteString("└──────────────┴──────────┴─────────┴─────┴──────────┘\n")
	return sb.String()
}

// FormatVAR 将 VAR 模型摘要格式化为文本输出。
func FormatVAR(summary VARSummary) string {
	var sb strings.Builder
	sb.WriteString("┌─────────────────────────────────────────┐\n")
	sb.WriteString("│        VAR 模型摘要 (VAR Summary)        │\n")
	sb.WriteString("├─────────────────────────────────────────┤\n")
	sb.WriteString(fmt.Sprintf("│ 模型阶数:     %d\n", summary.Order))
	sb.WriteString(fmt.Sprintf("│ 变量数:       %d\n", summary.NumVars))
	sb.WriteString(fmt.Sprintf("│ 有效观测数:   %d\n", summary.NumObs))
	sb.WriteString(fmt.Sprintf("│ AIC:          %.4f\n", summary.AIC))
	sb.WriteString(fmt.Sprintf("│ BIC:          %.4f\n", summary.BIC))
	sb.WriteString("├─────────────────────────────────────────┤\n")
	sb.WriteString("│ 各方程 R²:\n")

	for i, name := range summary.Variables {
		r2 := 0.0
		if i < len(summary.R2) {
			r2 = summary.R2[i]
		}
		sb.WriteString(fmt.Sprintf("│   %-12s  R² = %.4f\n", name, r2))
	}
	sb.WriteString("└─────────────────────────────────────────┘\n")
	return sb.String()
}

// FormatJSON 将任意结果对象格式化为 JSON 字符串。
// 支持缩进格式输出，便于阅读和调试。
func FormatJSON(v interface{}) (string, error) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return "", fmt.Errorf("JSON 序列化失败: %w", err)
	}
	return string(data), nil
}

// FormatJSONCompact 将结果格式化为紧凑 JSON（无缩进）。
// 适用于机器间数据传输。
func FormatJSONCompact(v interface{}) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("JSON 序列化失败: %w", err)
	}
	return string(data), nil
}

// truncate 截断字符串到指定长度
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen-1]) + "…"
}
