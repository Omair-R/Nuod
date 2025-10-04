package mdarray_tests

import "core:log"
import "core:testing"

@require import md "../../nuod/mdarray"


@(test)
test_permute_dims_view :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f32, [2]int{3, 4})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_dims_view(arr, [2]int{1, 0})

		testing.expect_value(t,  tr_arr.shape, [2]int{4, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect(t,  tr_arr.is_view)
		testing.expect_value(t,  tr_arr.strides, [2]int{1, 4})

		expected_arr := []f32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
		for i in 0..<12{
			testing.expect_value(t, md.get_linear(tr_arr, i), expected_arr[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_dims_view(arr, [3]int{0, 2, 1})

		testing.expect_value(t,  tr_arr.shape, [3]int{2, 2, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect(t,  tr_arr.is_view)
		testing.expect_value(t,  tr_arr.strides, [3]int{6, 1, 2})

		expected_arr := []f32{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11}
		for i in 0..<12{
			testing.expect_value(t, md.get_linear(tr_arr, i), expected_arr[i])
		}
	}
}


@(test)
test_permute_dims_copy :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f32, [2]int{3, 4})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_dims_copy(arr, [2]int{1, 0})
		defer md.free_mdarray(tr_arr)

		testing.expect_value(t,  tr_arr.shape, [2]int{4, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect_value(t,  tr_arr.is_view, false)
		testing.expect_value(t,  tr_arr.strides, [2]int{3, 1})

		expected_arr := []f32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
		for i in 0..<12{
			testing.expect_value(t, tr_arr.buffer[i], expected_arr[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_dims_copy(arr, [3]int{0, 2, 1})
		defer md.free_mdarray(tr_arr)

		testing.expect_value(t,  tr_arr.shape, [3]int{2, 2, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect_value(t,  tr_arr.is_view, false)
		testing.expect_value(t,  tr_arr.strides, [3]int{6, 3, 1})

		expected_arr := []f32{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11}
		for i in 0..<12{
			testing.expect_value(t, tr_arr.buffer[i], expected_arr[i])
		}
	}
}


@(test)
test_permute_default_view :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f32, [2]int{3, 4})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_default_view(arr)

		testing.expect_value(t,  tr_arr.shape, [2]int{4, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect(t,  tr_arr.is_view)
		testing.expect_value(t,  tr_arr.strides, [2]int{1, 4})

		expected_arr := []f32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
		for i in 0..<12{
			testing.expect_value(t, md.get_linear(tr_arr, i), expected_arr[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_default_view(arr)

		testing.expect_value(t,  tr_arr.shape, [3]int{2, 3, 2})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect(t,  tr_arr.is_view)
		testing.expect_value(t,  tr_arr.strides, [3]int{1, 2, 6})
	}
}


@(test)
test_permute_default_copy :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f32, [2]int{3, 4})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_default_copy(arr)
		defer md.free_mdarray(tr_arr)

		testing.expect_value(t,  tr_arr.shape, [2]int{4, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect_value(t,  tr_arr.is_view, false)
		testing.expect_value(t,  tr_arr.strides, [2]int{3, 1})

		expected_arr := []f32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
		for i in 0..<12{
			testing.expect_value(t, md.get_linear(tr_arr, i), expected_arr[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		tr_arr := md.permute_default_copy(arr)
		defer md.free_mdarray(tr_arr)

		testing.expect_value(t,  tr_arr.shape, [3]int{2, 3, 2})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect_value(t,  tr_arr.is_view, false)
		testing.expect_value(t,  tr_arr.strides, [3]int{6, 2, 1})
	}
}


@(test)
test_swap_axes_view :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f32, [2]int{3, 4})
		defer md.free_mdarray(arr)

		tr_arr := md.swap_axes_view(arr, 0, 1)

		testing.expect_value(t,  tr_arr.shape, [2]int{4, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect(t,  tr_arr.is_view)
		testing.expect_value(t,  tr_arr.strides, [2]int{1, 4})

		expected_arr := []f32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
		for i in 0..<12{
			testing.expect_value(t, md.get_linear(tr_arr, i), expected_arr[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		tr_arr := md.swap_axes_view(arr, 2, 1)

		testing.expect_value(t,  tr_arr.shape, [3]int{2, 2, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect(t,  tr_arr.is_view)
		testing.expect_value(t,  tr_arr.strides, [3]int{6, 1, 2})

		expected_arr := []f32{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11}
		for i in 0..<12{
			testing.expect_value(t, md.get_linear(tr_arr, i), expected_arr[i])
		}
	}
}


@(test)
test_swap_axes_copy :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f32, [2]int{3, 4})
		defer md.free_mdarray(arr)

		tr_arr := md.swap_axes_copy(arr, 0, 1)
		defer md.free_mdarray(tr_arr)

		testing.expect_value(t,  tr_arr.shape, [2]int{4, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect_value(t,  tr_arr.is_view, false)
		testing.expect_value(t,  tr_arr.strides, [2]int{3, 1})

		expected_arr := []f32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
		for i in 0..<12{
			testing.expect_value(t, tr_arr.buffer[i], expected_arr[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		tr_arr := md.swap_axes_copy(arr, 2, 1)
		defer md.free_mdarray(tr_arr)

		testing.expect_value(t,  tr_arr.shape, [3]int{2, 2, 3})
		testing.expect_value(t,  md.size(tr_arr), 12)
		testing.expect_value(t,  tr_arr.is_view, false)
		testing.expect_value(t,  tr_arr.strides, [3]int{6, 3, 1})

		expected_arr := []f32{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11}
		for i in 0..<12{
			testing.expect_value(t, tr_arr.buffer[i], expected_arr[i])
		}
	}
}
