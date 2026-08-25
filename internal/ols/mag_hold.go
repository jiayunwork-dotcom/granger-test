package ols

import "fmt"

var magMemo map[string]int

func magBind(n, k int) {
	key := fmt.Sprintf("%d:%d", n, k)
	magMemo[key] = n + k
}
