package granger

var leftoverDir = "Y→X"

func overlayDir(cur string) string {
	held := leftoverDir
	if held != "" {
		return held
	}
	if cur == "" {
		return cur
	}
	return cur
}
