package mdarray_tests

import "core:log"
import "core:testing"

@require import md "../../nuod/mdarray"


@(test)
test_make_array :: proc(t: ^testing.T){
	arr := md.make_mdarray(f32, [3]int{2, 3, 4})
	defer md.free_mdarray(arr)

	testing.expect_value(t,  arr.shape, [3]int{2, 3, 4})
	testing.expect_value(t,  len(arr.buffer), 24)
	testing.expect_value(t,  arr.is_view, false)
	testing.expect_value(t,  arr.strides, [3]int{12, 4, 1})

	
	arr2 := md.make_mdarray(int, [4]int{1, 1, 1, 1})
	defer md.free_mdarray(arr2)
	testing.expect_value(t,  arr2.shape, [4]int{1, 1, 1, 1})
	testing.expect_value(t,  len(arr2.buffer), 1)
	testing.expect_value(t,  arr2.strides, [4]int{1, 1, 1, 1})
}


@(test)
test_fills :: proc(t: ^testing.T){

	{
		arr := md.fills(f64(1.5), [3]int{2, 3, 4})
		defer md.free_mdarray(arr)


		testing.expect_value(t,  md.get_type(arr), f64)
		for i in 0..<md.size(arr){
			testing.expect_value(t,  arr.buffer[i], 1.5)
		}

		arr2 := md.fills_like(u32(3), arr)
		defer md.free_mdarray(arr2)
		
		testing.expect_value(t,  md.get_type(arr2), u32)
		for i in 0..<md.size(arr2){
			testing.expect_value(t,  arr2.buffer[i], 3)
		}
	}

	
	{
		arr := md.zeros(f32, [3]int{2, 3, 4})
		defer md.free_mdarray(arr)


		testing.expect_value(t,  md.get_type(arr), f32)
		for i in 0..<md.size(arr){
			testing.expect_value(t,  arr.buffer[i], 0.0)
		}

		arr2 := md.zeros_like(u64, arr)
		defer md.free_mdarray(arr2)
		
		testing.expect_value(t,  md.get_type(arr2), u64)
		for i in 0..<md.size(arr2){
			testing.expect_value(t,  arr2.buffer[i], 0)
		}
	}

	{
		arr := md.ones(f16, [3]int{2, 3, 4})
		defer md.free_mdarray(arr)


		testing.expect_value(t,  md.get_type(arr), f16)
		for i in 0..<md.size(arr){
			testing.expect_value(t,  arr.buffer[i], 1.0)
		}

		arr2 := md.ones_like(u16, arr)
		defer md.free_mdarray(arr2)
		
		testing.expect_value(t,  md.get_type(arr2), u16)
		for i in 0..<md.size(arr2){
			testing.expect_value(t,  arr2.buffer[i], 1)
		}
	}
}


@(test)
test_range:: proc(t: ^testing.T){	

	{
		arr := md.from_range(int, 4)
		defer md.free_mdarray(arr)

		testing.expect_value(t,  md.get_type(arr), int)
		testing.expect_value(t,  md.size(arr), 4)
		for i in 0..<4{
			testing.expect_value(t,  arr.buffer[i], int(i))
		}

		
		arr2 := md.from_range(i32, 6, 2)
		defer md.free_mdarray(arr2)

		testing.expect_value(t,  md.get_type(arr2), i32)
		testing.expect_value(t,  md.size(arr2), 4)
		arr_comp := []i32{2, 3, 4, 5}
		for a, i in arr_comp{
			testing.expect_value(t,  arr2.buffer[i], a)
		}

		arr3 := md.from_range(i128, 15, 2, 3)
		defer md.free_mdarray(arr3)

		testing.expect_value(t,  md.get_type(arr3), i128)
		testing.expect_value(t,  md.size(arr3), 5)
		arr_comp2 := []i128{2, 5, 8, 11, 14}
		for a, i in arr_comp2{
			testing.expect_value(t,  arr3.buffer[i], a)
		}
	}

	{
		arr := md.reshaped_range(f32, [3]int{2, 3, 4})
		defer md.free_mdarray(arr)

		testing.expect_value(t,  md.get_type(arr), f32)
		testing.expect_value(t,  md.size(arr), 24)
		for i in 0..<24{
			testing.expect_value(t,  arr.buffer[i], f32(i))
		}

		
		arr2 := md.reshaped_range(u64, [3]int{3, 4, 2}, begin=2)
		defer md.free_mdarray(arr2)

		testing.expect_value(t,  md.get_type(arr2), u64)
		testing.expect_value(t,  md.size(arr2), 24)
		for a, i in 2..<26{
			testing.expect_value(t,  arr2.buffer[i], u64(a))
		}

		
		arr3 := md.reshaped_range(i128, [2]int{3, 2}, begin=2, step=4)
		defer md.free_mdarray(arr3)

		testing.expect_value(t,  md.get_type(arr3), i128)
		testing.expect_value(t,  md.size(arr3), 6)
		arr_comp := []i128{2, 6, 10, 14, 18, 22}
		for a, i in arr_comp{
			testing.expect_value(t,  arr3.buffer[i], i128(a))
		}
	}
}


@(test)
test_space:: proc(t: ^testing.T){
	EPS :: 0.00001

	{
		arr := md.linspace(f32, 0, 5, 6)
		defer md.free_mdarray(arr)

		testing.expect_value(t,  md.get_type(arr), f32)
		testing.expect_value(t,  md.size(arr), 6)
		for i in 0..<6{
			testing.expect_value(t,  arr.buffer[i], f32(i))
		}

		arr2 := md.linspace(f64, 1.0, 2.0, 4)
		defer md.free_mdarray(arr2)

		testing.expect_value(t,  md.get_type(arr2), f64)
		testing.expect_value(t,  md.size(arr2), 4)
		arr_comp := []f64{1.0, 1.333333333, 1.66666666667, 2.0}
		for a, i in arr_comp{
			testing.expect(t,  abs(arr2.buffer[i] - f64(a)) <= EPS)
		}
	}

	
	{
		arr := md.logspace(f64, 2.0, 3.0, 4)
		defer md.free_mdarray(arr)

		testing.expect_value(t,  md.get_type(arr), f64)
		testing.expect_value(t,  md.size(arr), 4)
		arr_comp := []f64{100.0, 215.443469003, 464.1588833612, 1000.0}
		for a, i in arr_comp{
			testing.expect(t,  abs(arr.buffer[i] - f64(a)) <= EPS)
		}

		arr2 := md.logspace(f64, 2.0, 3.0, 4, endpoint=false)
		defer md.free_mdarray(arr2)

		testing.expect_value(t,  md.get_type(arr2), f64)
		testing.expect_value(t,  md.size(arr2), 4)
		arr_comp2 := []f64{100.0, 177.827941, 316.22776602, 562.34132519}
		for a, i in arr_comp2{
			testing.expect(t,  abs(arr2.buffer[i] - f64(a)) <= EPS)
		}

		arr3 := md.logspace(f64, 2.0, 3.0, 4, base=2.0)
		defer md.free_mdarray(arr3)

		testing.expect_value(t,  md.get_type(arr3), f64)
		testing.expect_value(t,  md.size(arr3), 4)
		arr_comp3 := []f64{4.0, 5.0396842 , 6.34960421, 8.0}
		for a, i in arr_comp3{
			testing.expect(t,  abs(arr3.buffer[i] - a) <= EPS)
		}
	}
}


@(test)
test_eye :: proc(t: ^testing.T){

	{
		arr := md.eye(int, 2, 3)
		defer md.free_mdarray(arr)
			
		testing.expect_value(t,  md.get_type(arr), int)
		testing.expect_value(t,  md.size(arr), 6)

		arr_comp := []int{1, 0, 0, 0, 1, 0}
		for a, i in arr_comp{
			testing.expect_value(t,  arr.buffer[i], a)
		}
		
		arr2 := md.eye(int, 3, 3, diag_idx=1)
		defer md.free_mdarray(arr2)
			
		testing.expect_value(t,  md.get_type(arr2), int)
		testing.expect_value(t,  md.size(arr2), 9)

		arr_comp2 := []int{0, 1, 0, 0, 0, 1, 0, 0, 0}
		for a, i in arr_comp2{
			testing.expect_value(t,  arr2.buffer[i], a)
		}
		
		arr3 := md.eye(int, 3, 3, diag_idx=-1)
		defer md.free_mdarray(arr3)
			
		testing.expect_value(t,  md.get_type(arr3), int)
		testing.expect_value(t,  md.size(arr3), 9)

		arr_comp3 := []int{0, 0, 0, 1, 0, 0, 0, 1, 0}
		for a, i in arr_comp3{
			testing.expect_value(t,  arr3.buffer[i], a)
		}
		
		arr4 := md.eye(int, 3, 3, diag_idx=-2)
		defer md.free_mdarray(arr4)
			
		testing.expect_value(t,  md.get_type(arr4), int)
		testing.expect_value(t,  md.size(arr4), 9)

		arr_comp4 := []int{0, 0, 0, 0, 0, 0, 1, 0, 0}
		for a, i in arr_comp4{
			testing.expect_value(t,  arr4.buffer[i], a)
		}
	}
	
	{
		arr := md.identity(int, 3)
		defer md.free_mdarray(arr)

			
		testing.expect_value(t,  md.get_type(arr), int)
		testing.expect_value(t,  md.size(arr), 9)

		arr_comp := []int{1, 0, 0, 0, 1, 0, 0 , 0, 1}
		for a, i in arr_comp{
			testing.expect_value(t,  arr.buffer[i], a)
		}

		
		arr2 := md.identity(i128, 4)
		defer md.free_mdarray(arr2)

			
		testing.expect_value(t,  md.get_type(arr2), i128)
		testing.expect_value(t,  md.size(arr2), 16)

		arr_comp2 := []i128{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		for a, i in arr_comp2{
			testing.expect_value(t,  arr2.buffer[i], a)
		}
	}
}


@(test)
test_copy_cast :: proc(t: ^testing.T){

	{
		arr := md.from_range(f32, 4)
		defer md.free_mdarray(arr)

		arr2 := md.copy_array(arr)
		defer md.free_mdarray(arr2)

		testing.expect_value(t,  md.get_type(arr2), f32)
		testing.expect_value(t,  md.size(arr2), 4)
		for i in 0..<4{
			testing.expect_value(t,  arr.buffer[i], arr2.buffer[i])
		}
	}
	
	{
		arr := md.from_range(f32, 4)
		defer md.free_mdarray(arr)

		arr2 := md.cast_array(arr, int)
		defer md.free_mdarray(arr2)

		testing.expect_value(t,  md.get_type(arr2), int)
		testing.expect_value(t,  md.size(arr2), 4)

		for i in 0..<4{
			testing.expect_value(t,  int(arr.buffer[i]), arr2.buffer[i])
		}
		
		arr3 := md.linspace(f64, 0.0, 1.0, 4)
		defer md.free_mdarray(arr3)

		arr4 := md.cast_array(arr3, int)
		defer md.free_mdarray(arr4)

		testing.expect_value(t,  md.get_type(arr4), int)
		testing.expect_value(t,  md.size(arr4), 4)
		arr_comp := []int{0, 0, 0, 1}
		for a, i in arr_comp{
			testing.expect_value(t,  arr4.buffer[i], a)
		}
	}
}


@(test)
test_reshape :: proc(t : ^testing.T){
		
	{
		arr := md.from_range(f32, 4)
		defer md.free_mdarray(arr)

		arr2 := md.reshape_view(arr, [2]int{2, 2})
		testing.expect_value(t,  md.get_type(arr2), f32)
		testing.expect_value(t,  md.size(arr2), 4)
		testing.expect(t,  arr2.is_view)
		testing.expect_value(t,  arr2.shape, [2]int{2, 2})
		testing.expect_value(t,  arr2.strides, [2]int{2, 1})

		// TODO figure out a way to test for this without test failure
		// arr3, ok := md.reshape_view(arr, [2]int{3,1})
		// testing.expect_value(t,  ok, false)
		//
		arr3 := md.reshape_view(arr, [2]int{2,-1})
		testing.expect_value(t,  md.get_type(arr3), f32)
		testing.expect_value(t,  md.size(arr3), 4)
		testing.expect(t,  arr3.is_view)
		testing.expect_value(t,  arr3.shape, [2]int{2, 2})
		testing.expect_value(t,  arr3.strides, [2]int{2, 1})
	}
	
	{
		arr := md.from_range(f64, 12)
		defer md.free_mdarray(arr)

		arr2 := md.reshape_copy(arr, [3]int{3, 2, 2})
		defer md.free_mdarray(arr2)
		testing.expect_value(t,  md.get_type(arr2), f64)
		testing.expect_value(t,  md.size(arr2), 12)
		testing.expect_value(t,  arr2.is_view, false)
		testing.expect_value(t,  arr2.shape, [3]int{3, 2, 2})
		testing.expect_value(t,  arr2.strides, [3]int{4, 2, 1})

		arr3 := md.reshape_copy(arr, [2]int{3,-1})
		defer md.free_mdarray(arr3)

		testing.expect_value(t,  md.get_type(arr3), f64)
		testing.expect_value(t,  md.size(arr3), 12)
		testing.expect_value(t,  arr3.is_view, false)
		testing.expect_value(t,  arr3.shape, [2]int{3, 4})
		testing.expect_value(t,  arr3.strides, [2]int{4, 1})
	}
}


@(test)
test_expand_dim :: proc(t : ^testing.T){
		
	{
		arr := md.reshaped_range(f32, [2]int{2, 2})
		defer md.free_mdarray(arr)

		arr2 := md.expand_dim_view(2,arr, axis=0)
		testing.expect_value(t,  arr2.shape, [3]int{1, 2, 2})
		testing.expect_value(t,  md.get_type(arr2), f32)
		testing.expect_value(t,  md.size(arr2), 4)
		testing.expect(t,  arr2.is_view)
		testing.expect_value(t,  arr2.strides, [3]int{4, 2, 1})

		arr3 := md.expand_dim_view(2, arr, axis=1)
		testing.expect_value(t,  arr3.shape, [3]int{2, 1, 2})
		arr4 := md.expand_dim_view(2, arr, axis=2)
		testing.expect_value(t,  arr4.shape, [3]int{2, 2, 1})
	}
	
	{
		arr := md.reshaped_range(f32, [2]int{2, 2})
		defer md.free_mdarray(arr)

		arr2 := md.expand_dim_copy(2,arr, axis=0)
		defer md.free_mdarray(arr2)
		testing.expect_value(t,  arr2.shape, [3]int{1, 2, 2})
		testing.expect_value(t,  md.get_type(arr2), f32)
		testing.expect_value(t,  len(arr2.buffer), 4)
		testing.expect_value(t,  arr2.is_view, false)
		testing.expect_value(t,  arr2.strides, [3]int{4, 2, 1})

		arr3 := md.expand_dim_copy(2, arr, axis=1)
		defer md.free_mdarray(arr3)
		testing.expect_value(t,  arr3.shape, [3]int{2, 1, 2})
		testing.expect_value(t,  len(arr3.buffer), 4)
	}
}


@(test)
test_flatten :: proc(t : ^testing.T){
	{
		arr := md.reshaped_range(f64, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		arr2 := md.flatten_view(arr)
		
		testing.expect_value(t,  arr2.shape, [1]int{12})
		testing.expect_value(t,  md.get_type(arr2), f64)
		testing.expect_value(t,  md.size(arr2), 12)
		testing.expect(t,  arr2.is_view)
		testing.expect_value(t,  arr2.strides, [1]int{1})
	}
	
	{
		arr := md.reshaped_range(f64, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)

		arr2 := md.flatten_copy(arr)
		defer md.free_mdarray(arr2)
				
		testing.expect_value(t,  arr2.shape, [1]int{12})
		testing.expect_value(t,  md.get_type(arr2), f64)
		testing.expect_value(t,  len(arr2.buffer), 12)
		testing.expect_value(t,  arr2.is_view, false)
		testing.expect_value(t,  arr2.strides, [1]int{1})

		for i in 0..<12 {
			testing.expect_value(t, arr2.buffer[i], f64(i))
		}
	}
}


@(test)
test_broadcast_to :: proc(t : ^testing.T){

	{
		arr := md.reshaped_range(f64, [2]int{1, 4})
		defer md.free_mdarray(arr)

		br_arr := md.broadcast_to(arr, [2]int{3, 4})
		defer md.free_mdarray(br_arr)

		testing.expect_value(t, md.size(br_arr), 12)
		testing.expect_value(t, br_arr.is_view, false)
		testing.expect_value(t, br_arr.shape, [2]int{3, 4})
		
		expected_br := []f64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}
		for i in 0..<12{
			testing.expect_value(t, br_arr.buffer[i], expected_br[i])
		}
	}
	
	{
		arr := md.reshaped_range(f64, [1]int{4})
		defer md.free_mdarray(arr)

		br_arr := md.broadcast_to(arr, [2]int{3, 4})
		defer md.free_mdarray(br_arr)

		testing.expect_value(t, md.size(br_arr), 12)
		testing.expect_value(t, br_arr.is_view, false)
		testing.expect_value(t, br_arr.shape, [2]int{3, 4})
		
		expected_br := []f64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}
		for i in 0..<12{
			testing.expect_value(t, br_arr.buffer[i], expected_br[i])
		}
	}

	{
		arr := md.reshaped_range(int, [3]int{1, 2, 1})
		defer md.free_mdarray(arr)

		br_arr := md.broadcast_to(arr, [3]int{2, 2, 3})
		defer md.free_mdarray(br_arr)

		testing.expect_value(t, md.size(br_arr), 12)
		testing.expect_value(t, br_arr.is_view, false)
		testing.expect_value(t, br_arr.shape, [3]int{2, 2, 3})
		
		expected_br := []int{0, 0, 0,  1, 1, 1, 0, 0, 0,  1, 1, 1}
		for i in 0..<12{
			testing.expect_value(t, br_arr.buffer[i], expected_br[i])
		}
	}
}


@(test)
test_broadcast_map :: proc(t : ^testing.T){

	make_f :: proc($T: typeid) -> proc(T, T, ..T) -> T {
		return proc(a: T, b: T, args: ..T) -> T {
			return a * b
		}
	}

	{
		arr := md.reshaped_range(f64, [2]int{1, 4})
		defer md.free_mdarray(arr)

		arr2 := md.reshaped_range(f64, [2]int{3, 1})
		defer md.free_mdarray(arr2)

		br_arr := md.broadcast_map(arr, arr2, make_f(f64))
		defer md.free_mdarray(br_arr)

		testing.expect_value(t, md.size(br_arr), 12)
		testing.expect_value(t, br_arr.is_view, false)
		testing.expect_value(t, br_arr.shape, [2]int{3, 4})
		
		expected_br := []f64{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6}
		for i in 0..<12{
			testing.expect_value(t, br_arr.buffer[i], expected_br[i])
		}
	}
	
	{
		arr := md.reshaped_range(f32, [3]int{1, 2, 1})
		defer md.free_mdarray(arr)

		arr2 := md.reshaped_range(f32, [3]int{2, 1, 3})
		defer md.free_mdarray(arr2)

		br_arr := md.broadcast_map(arr, arr2, make_f(f32))
		defer md.free_mdarray(br_arr)

		testing.expect_value(t, md.size(br_arr), 12)
		testing.expect_value(t, br_arr.is_view, false)
		testing.expect_value(t, br_arr.shape, [3]int{2, 2, 3})
		
		expected_br := []f32{0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 5}
		for i in 0..<12{
			testing.expect_value(t, br_arr.buffer[i], expected_br[i])
		}
	}
}


@(test)
test_stack :: proc(t : ^testing.T){
	{
		arr := md.reshaped_range(f64, [2]int{1, 4})
		defer md.free_mdarray(arr)

		arr2 := md.reshaped_range(f64, [2]int{1, 4})
		defer md.free_mdarray(arr2)

		st_arr := md.stack(2, []md.MdArray(f64, 2){arr, arr2}, )
		defer md.free_mdarray(st_arr)

		testing.expect_value(t, md.size(st_arr), 8)
		testing.expect_value(t, st_arr.is_view, false)
		testing.expect_value(t, st_arr.shape, [3]int{2, 1, 4})

		expected_st := []f64{0, 1, 2, 3, 0, 1, 2, 3}
		for i in 0..<8{
			testing.expect_value(t, st_arr.buffer[i], expected_st[i])
		}
	}
	
	{
		arr := md.reshaped_range(f64, [2]int{2, 3})
		defer md.free_mdarray(arr)

		arr2 := md.reshaped_range(f64, [2]int{2, 3})
		defer md.free_mdarray(arr2)

		st_arr := md.stack(2, []md.MdArray(f64, 2){arr, arr2}, axis=1)
		defer md.free_mdarray(st_arr)

		testing.expect_value(t, md.size(st_arr), 12)
		testing.expect_value(t, st_arr.is_view, false)
		testing.expect_value(t, st_arr.shape, [3]int{2, 2, 3})

		expected_st := []f64{0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5}
		for i in 0..<12{
			testing.expect_value(t, st_arr.buffer[i], expected_st[i])
		}
	}
}


@(test)
test_vstack :: proc(t : ^testing.T){
	arr := md.reshaped_range(f64, [1]int{4})
	defer md.free_mdarray(arr)

	arr2 := md.reshaped_range(f64, [1]int{4})
	defer md.free_mdarray(arr2)

	arr3 := md.reshaped_range(f64, [1]int{4})
	defer md.free_mdarray(arr3)

	st_arr := md.vstack([]md.MdArray(f64, 1){arr, arr2, arr3})
	defer md.free_mdarray(st_arr)

	testing.expect_value(t, md.size(st_arr), 12)
	testing.expect_value(t, st_arr.is_view, false)
	testing.expect_value(t, st_arr.shape, [2]int{3, 4})

	expected_st := []f64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}
	for i in 0..<12{
		testing.expect_value(t, st_arr.buffer[i], expected_st[i])
	}
}


@(test)
test_hstack :: proc(t : ^testing.T){
	arr := md.reshaped_range(f64, [1]int{4})
	defer md.free_mdarray(arr)

	arr2 := md.reshaped_range(f64, [1]int{4})
	defer md.free_mdarray(arr2)

	st_arr := md.hstack([]md.MdArray(f64, 1){arr, arr2})
	defer md.free_mdarray(st_arr)

	testing.expect_value(t, md.size(st_arr), 8)
	testing.expect_value(t, st_arr.is_view, false)
	testing.expect_value(t, st_arr.shape, [1]int{8})

	expected_st := []f64{0, 1, 2, 3, 0, 1, 2, 3}
	for i in 0..<8{
		testing.expect_value(t, st_arr.buffer[i], expected_st[i])
	}
}


@(test)
test_concat :: proc(t : ^testing.T){
	arr := md.reshaped_range(f64, [2]int{2, 3})
	defer md.free_mdarray(arr)

	arr2 := md.reshaped_range(f64, [2]int{1, 3})
	defer md.free_mdarray(arr2)

	st_arr := md.concat([]md.MdArray(f64, 2){arr, arr2})
	defer md.free_mdarray(st_arr)

	testing.expect_value(t, md.size(st_arr), 9)
	testing.expect_value(t, st_arr.is_view, false)
	testing.expect_value(t, st_arr.shape, [2]int{3, 3})

	expected_st := []f64{0, 1, 2, 3, 4, 5, 0, 1, 2}
	for i in 0..<9{
		testing.expect_value(t, st_arr.buffer[i], expected_st[i])
	}
}


@(test)
test_where_cond :: proc(t : ^testing.T){
	arr := md.reshaped_range(f64, [1]int{8})
	defer md.free_mdarray(arr)

	mask := md.from_slice([]bool{
		false, false, true, true, false, false, true, true
	}, [1]int{8})

	defer md.free_mdarray(mask)

	masked_arr := md.where_cond(arr, mask)
	defer md.free_mdarray(masked_arr)

	testing.expect_value(t, md.size(masked_arr), 4)
	testing.expect_value(t, masked_arr.is_view, false)
	testing.expect_value(t, masked_arr.shape, [1]int{4})

	expected_arr := []f64{2, 3, 6, 7}
	for i in 0..<4{
		testing.expect_value(t, masked_arr.buffer[i], expected_arr[i])
	}
}
