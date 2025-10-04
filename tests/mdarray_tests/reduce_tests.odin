package mdarray_tests

import "core:testing"

@require import md "../../nuod/mdarray"


@private
make_sum_f :: proc($T: typeid)-> proc(T, T,..T)->T{
	return proc (accum: T, val: T, args: ..T) -> T { return accum + val}
}


@(test)
test_all_reduce :: proc(t: ^testing.T){
	arr := md.reshaped_range(f32, [2]int{2, 3})
	defer md.free_mdarray(arr)

	reduced_val := md.all_reduce_map(arr, make_sum_f(f32), f32(0))
	testing.expect_value(t, reduced_val, 15)

	reduced_val = md.all_reduce_sum(arr)
	testing.expect_value(t, reduced_val, 15)

	reduced_val = md.all_reduce_prod(arr)
	testing.expect_value(t, reduced_val, 0)

	reduced_val = md.all_reduce_max(arr)
	testing.expect_value(t, reduced_val, 5)

	reduced_val = md.all_reduce_min(arr)
	testing.expect_value(t, reduced_val, 0)

	reduced_val = md.all_reduce_avg(arr)
	testing.expect_value(t, reduced_val, 2.5)
}


@(test)
test_dim_reduce :: proc(t: ^testing.T){
	arr := md.reshaped_range(f32, [2]int{2, 3})
	defer md.free_mdarray(arr)

	red_arr := md.dim_reduce_map(2, arr, 0, make_sum_f(f32), f32(0))
	defer md.free_mdarray(red_arr)

	testing.expect_value(t, red_arr.shape, [1]int{3})
	testing.expect_value(t, len(red_arr.buffer), 3)

	expected_arr := []f32{3, 5, 7}
	for i in 0..<md.size(red_arr){
		testing.expect_value(t, red_arr.buffer[i], expected_arr[i])
	}

	
	red_arr2 := md.dim_reduce_prod(2, arr, 1)
	defer md.free_mdarray(red_arr2)

	testing.expect_value(t, red_arr2.shape, [1]int{2})
	testing.expect_value(t, len(red_arr2.buffer), 2)

	expected_arr2 := []f32{0, 60}
	for i in 0..<md.size(red_arr2){
		testing.expect_value(t, red_arr2.buffer[i], expected_arr2[i])
	}

	red_arr3 := md.dim_reduce_avg(2, arr, 1)
	defer md.free_mdarray(red_arr3)

	expected_arr3 := []f32{1, 4}
	for i in 0..<md.size(red_arr3){
		testing.expect_value(t, red_arr3.buffer[i], expected_arr3[i])
	}
}


@(test)
test_keepdim_reduce :: proc(t: ^testing.T){
	arr := md.reshaped_range(f32, [2]int{2, 3})
	defer md.free_mdarray(arr)

	red_arr := md.keepdim_reduce_map(2, arr, 0, make_sum_f(f32), f32(0))
	defer md.free_mdarray(red_arr)

	testing.expect_value(t, red_arr.shape, [2]int{1, 3})
	testing.expect_value(t, len(red_arr.buffer), 3)

	expected_arr := []f32{3, 5, 7}
	for i in 0..<md.size(red_arr){
		testing.expect_value(t, red_arr.buffer[i], expected_arr[i])
	}

	red_arr2 := md.keepdim_reduce_prod(arr, 1)
	defer md.free_mdarray(red_arr2)

	testing.expect_value(t, red_arr2.shape, [2]int{2, 1})
	testing.expect_value(t, len(red_arr2.buffer), 2)

	expected_arr2 := []f32{0, 60}
	for i in 0..<md.size(red_arr2){
		testing.expect_value(t, red_arr2.buffer[i], expected_arr2[i])
	}
}
