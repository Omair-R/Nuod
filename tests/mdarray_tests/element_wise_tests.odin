package mdarray_tests

import "core:testing"

@require import md "../../nuod/mdarray"


@(private="file")
make_sum_f :: proc($T: typeid)-> proc(T, T,..T)->T{
	return proc (a: T, b: T, args: ..T) -> T { return a + b }
}


@(private="file")
make_with_args_f :: proc($T: typeid)-> proc(T, T,..T)->T{
	return proc (a: T, b: T, args: ..T) -> T { return (a + b)/args[0] }
}


@(private="file")
make_div_f :: proc($T: typeid)-> proc(T, T,..T)->T{
	return proc (a: T, b: T, args: ..T) -> T { return a / b }
}

@(test)
test_element_wise_map :: proc(t: ^testing.T){

	arr := md.reshaped_range(f32, [2]int{2, 3})
	defer md.free_mdarray(arr)

	arr2 := md.reshaped_range(f32, [2]int{2, 3})
	defer md.free_mdarray(arr2)

	{
		el_arr, ok := md.element_wise_map(arr, arr2, make_sum_f(f32))
		testing.expect(t, ok)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, el_arr.shape, arr.shape)
		testing.expect_value(t, len(el_arr.buffer), len(arr.buffer))

		expect_arr := []f32{0, 2, 4, 6, 8, 10}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr, ok := md.element_wise_map(arr, arr2, make_with_args_f(f32), f32(2))
		testing.expect(t, ok)
		defer md.free_mdarray(el_arr)

		expect_arr := []f32{0, 1, 2, 3, 4, 5}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		
		f :: proc (a: f32, b: f32, args: ..int) -> int { return int(a + b)/args[0] }
		el_arr, ok := md.element_wise_map(arr, arr2, f, int(2))
		testing.expect(t, ok)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, md.get_type(el_arr), int)

		expect_arr := []int{0, 1, 2, 3, 4, 5}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr := md.add(arr, arr2)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, el_arr.shape, arr.shape)
		testing.expect_value(t, len(el_arr.buffer), len(arr.buffer))

		expect_arr := []f32{0, 2, 4, 6, 8, 10}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr := md.mul(arr, arr2)
		defer md.free_mdarray(el_arr)

		expect_arr := []f32{0, 1, 4, 9, 16, 25}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}
}

@(test)
test_scalar_map :: proc(t: ^testing.T){

	arr := md.reshaped_range(f32, [2]int{2, 3})
	defer md.free_mdarray(arr)

	scal := f32(3)
	
	{
		el_arr, ok := md.scalar_map(arr, scal, make_sum_f(f32))
		testing.expect(t, ok)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, el_arr.shape, arr.shape)
		testing.expect_value(t, len(el_arr.buffer), len(arr.buffer))

		expect_arr := []f32{3, 4, 5, 6, 7, 8}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr := md.add(arr, scal)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, el_arr.shape, arr.shape)
		testing.expect_value(t, len(el_arr.buffer), len(arr.buffer))

		expect_arr := []f32{3, 4, 5, 6, 7, 8}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr := md.mul(arr, scal)
		defer md.free_mdarray(el_arr)

		expect_arr := []f32{0, 3, 6, 9, 12, 15}
		for i in 0..<6 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}
}


@(test)
test_flip_scalar_map :: proc(t: ^testing.T){

	arr := md.from_slice([]f32{1, 2, 1, 2}, [2]int{2, 2})
	defer md.free_mdarray(arr)

	scal := f32(3)
	
	{
		el_arr, ok := md.scalar_map(arr, scal, make_div_f(f32), flip=true)
		testing.expect(t, ok)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, el_arr.shape, arr.shape)
		testing.expect_value(t, len(el_arr.buffer), len(arr.buffer))

		expect_arr := []f32{3, 1.5, 3, 1.5}
		for i in 0..<4 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr := md.add(scal, arr)
		defer md.free_mdarray(el_arr)

		testing.expect_value(t, el_arr.shape, arr.shape)
		testing.expect_value(t, len(el_arr.buffer), len(arr.buffer))

		expect_arr := []f32{4, 5, 4, 5}
		for i in 0..<4 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}

	{
		el_arr := md.div(scal, arr)
		defer md.free_mdarray(el_arr)

		expect_arr := []f32{3, 1.5, 3, 1.5}
		for i in 0..<4 {
			testing.expect_value(t, el_arr.buffer[i], expect_arr[i])
		}
	}
}
