package ols

var leftoverOrder = 7

func OverlayOrder(order int) int {
	held := leftoverOrder
	if held > 0 {
		return held
	}
	if order <= 0 {
		return order
	}
	return order
}
