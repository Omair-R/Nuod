package mdarray_tests

import "core:testing"

@require import md "../../nuod/mdarray"
@require import nl "../../nuod/linalg"


@(test)
test_diagonal :: proc(t: ^testing.T){

	arr := md.reshaped_range(f32, [2]int{3, 3})
	defer md.free_mdarray(arr)

	{
		diag_arr := nl.matrix_diagonal(arr)
		defer md.free_mdarray(diag_arr)

		testing.expect_value(t, len(diag_arr.buffer), 3)
		testing.expect_value(t, diag_arr.shape, [1]int{3})

		expect_arr := []f32{0, 4, 8}
		for i in 0..<3{
			testing.expect_value(t, diag_arr.buffer[i], expect_arr[i])
		}
	}
	{
		diag_arr := nl.matrix_diagonal(arr, offset=1)
		defer md.free_mdarray(diag_arr)

		testing.expect_value(t, len(diag_arr.buffer), 2)
		testing.expect_value(t, diag_arr.shape, [1]int{2})

		expect_arr := []f32{1, 5}
		for i in 0..<2{
			testing.expect_value(t, diag_arr.buffer[i], expect_arr[i])
		}
	}
	{
		diag_arr := nl.matrix_diagonal(arr, offset=-2)
		defer md.free_mdarray(diag_arr)

		testing.expect_value(t, len(diag_arr.buffer), 1)
		testing.expect_value(t, diag_arr.shape, [1]int{1})

		testing.expect_value(t, diag_arr.buffer[0], 6)
	}

	
	{
		testing.expect_value(t, nl.matrix_trace(arr), f32(12))
		testing.expect_value(t, nl.matrix_trace(arr, offset=1), f32(6))
		testing.expect_value(t, nl.matrix_trace(arr, offset=-2), f32(6))
	}
}
