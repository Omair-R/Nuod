package linalg_tests


import "core:math"
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


@(test)
test_full_vector_norm :: proc(t: ^testing.T){

	arr := md.reshaped_range(f32, [1]int{5})
	defer md.free_mdarray(arr)

	val :f32 = nl.full_vector_norm(arr, .Euclidean)
	testing.expect_value(t, val, math.sqrt(f32(30)))

	val = nl.full_vector_norm(arr, .Manhattan)
	testing.expect_value(t, val, 10)

	val = nl.full_vector_norm(arr, .Chebyshev)
	testing.expect_value(t, val, 4)

	val = nl.full_vector_norm(arr, .L0)
	testing.expect_value(t, val, 4)
}


@(test)
test_dim_vector_norm :: proc(t: ^testing.T){

	arr := md.reshaped_range(f32, [2]int{3, 3})
	defer md.free_mdarray(arr)

	{
		norm_arr := nl.dim_vector_norm(2, arr, 0, .L2)
		defer md.free_mdarray(norm_arr)

		testing.expect_value(t, norm_arr.shape, [1]int{3})
		testing.expect_value(t, len(norm_arr.buffer), 3)

		expect_arr := []f32{math.sqrt(f32(5)), math.sqrt(f32(50)), math.sqrt(f32(147.0))}
		for i in 0..<3{
			testing.expect_value(t, len(norm_arr.buffer), 3)
		}
	}

	{
		norm_arr := nl.dim_vector_norm(2, arr, 0, .L1)
		defer md.free_mdarray(norm_arr)

		testing.expect_value(t, norm_arr.shape, [1]int{3})
		testing.expect_value(t, len(norm_arr.buffer), 3)

		expect_arr := []f32{2, 12, 21}
		for i in 0..<3{
			testing.expect_value(t, len(norm_arr.buffer), 3)
		}
	}

	{
		norm_arr := nl.dim_vector_norm(2, arr, 1, .Linfty)
		defer md.free_mdarray(norm_arr)

		testing.expect_value(t, norm_arr.shape, [1]int{3})
		testing.expect_value(t, len(norm_arr.buffer), 3)

		expect_arr := []f32{6, 7, 8}
		for i in 0..<3{
			testing.expect_value(t, len(norm_arr.buffer), 3)
		}
	}
}


