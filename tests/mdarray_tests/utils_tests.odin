package mdarray_tests

import "core:testing"

@require import md "../../nuod/mdarray"

@(test)
test_compute_strides :: proc(t: ^testing.T){
	testing.expect_value(t,  md.compute_strides([4]int{1, 2, 3, 4}), [4]int{24, 12, 4, 1})
	testing.expect_value(t, md.compute_strides([1]int{4}), [1]int{1})
}
