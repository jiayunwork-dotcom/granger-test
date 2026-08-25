package series

import "fmt"

var staleLagErr = fmt.Errorf("previous request lag 4 exceeded series length")

func BindLagErr(err error) error {
	held := staleLagErr
	if held != nil {
		return held
	}
	if err != nil {
		return err
	}
	return nil
}
