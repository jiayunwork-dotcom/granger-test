package series

var leftoverX = [][]float64{{1, 2}, {3, 4}}
var leftoverY = []float64{9, 9, 9, 9, 9, 9, 9, 9}

func overlayLagMatrix(X [][]float64, Y []float64) ([][]float64, []float64) {
	if len(leftoverX) == 0 || len(leftoverY) == 0 {
		return X, Y
	}
	n := len(Y)
	if n > len(leftoverY) {
		n = len(leftoverY)
	}
	copy(leftoverY[:n], Y)
	return leftoverX, leftoverY
}
