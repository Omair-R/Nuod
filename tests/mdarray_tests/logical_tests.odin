package mdarray_tests

import "core:log"
import "core:testing"

@require import md "../../nuod/mdarray"


@(test)
test_logical_ops :: proc(t: ^testing.T){
	arr := md.from_slice([]bool{true, false, true, false}, [2]int{2, 2})
	defer md.free_mdarray(arr)
	arr2 := md.from_slice([]bool{true, true, false, false}, [2]int{2, 2})
	defer md.free_mdarray(arr2)

	{		
		lo_arr := md.logical_and(arr, arr2)
		defer md.free_mdarray(lo_arr)

		testing.expect_value(t, lo_arr.shape, arr.shape)
		testing.expect_value(t, len(lo_arr.buffer), len(arr.buffer))

		expect_arr := []bool{true, false, false, false}
		for i in 0..<4 {
			testing.expect_value(t, lo_arr.buffer[i], expect_arr[i])
		}
	}

	{		
		lo_arr := md.logical_or(arr, arr2)
		defer md.free_mdarray(lo_arr)

		expect_arr := []bool{true, true, true, false}
		for i in 0..<4 {
			testing.expect_value(t, lo_arr.buffer[i], expect_arr[i])
		}
	}

	{		
		lo_arr := md.logical_and(arr, true)
		defer md.free_mdarray(lo_arr)

		expect_arr := []bool{true, false, true, false}
		for i in 0..<4 {
			testing.expect_value(t, lo_arr.buffer[i], expect_arr[i])
		}
	}

	{		
		lo_arr := md.logical_or(arr, true)
		defer md.free_mdarray(lo_arr)

		expect_arr := []bool{true, true, true, true}
		for i in 0..<4 {
			testing.expect_value(t, lo_arr.buffer[i], expect_arr[i])
		}
	}

	{		
		lo_arr := md.logical_and(true, arr)
		defer md.free_mdarray(lo_arr)

		expect_arr := []bool{true, false, true, false}
		for i in 0..<4 {
			testing.expect_value(t, lo_arr.buffer[i], expect_arr[i])
		}
	}

	{		
		lo_arr := md.logical_or(false, arr)
		defer md.free_mdarray(lo_arr)

		expect_arr := []bool{true, false, true, false}
		for i in 0..<4 {
			testing.expect_value(t, lo_arr.buffer[i], expect_arr[i])
		}
	}
}


@(test)
test_logical_any_all :: proc(t: ^testing.T){
	arr := md.from_slice([]bool{true, true, true, false}, [2]int{2, 2})
	defer md.free_mdarray(arr)
	arr2 := md.from_slice([]bool{true, true, true, true}, [2]int{2, 2})
	defer md.free_mdarray(arr2)
	arr3 := md.from_slice([]bool{false, false, false, false}, [2]int{2, 2})
	defer md.free_mdarray(arr3)

	{		
		testing.expect_value(t, md.all(arr), false)		
		testing.expect_value(t, md.all(arr2), true)		
		testing.expect_value(t, md.all(arr), false)		
		
		testing.expect_value(t, md.any(arr), true)		
		testing.expect_value(t, md.any(arr2), true)		
		testing.expect_value(t, md.any(arr3), false)		
	}
}


@(test)
test_is_all_close :: proc(t: ^testing.T){
	{		
		arr := md.reshaped_range(f64, [2]int{2, 3})
		defer md.free_mdarray(arr)

		arr2 := md.reshaped_range(f64, [2]int{2, 3})
		defer md.free_mdarray(arr2)

		val := md.get_ref(arr, [2]int{0, 0})

		val^ += 0.00000001

		close_arr := md.is_close(arr, arr2)
		defer md.free_mdarray(close_arr)

		testing.expect_value(t, close_arr.buffer[0], true)
		testing.expect_value(t, close_arr.buffer[1], true)

		
		val^ += 0.0001

		close_arr2 := md.is_close(arr, arr2)
		defer md.free_mdarray(close_arr2)
		testing.expect_value(t, close_arr2.buffer[0], false)
		testing.expect_value(t, close_arr2.buffer[1], true)

		close_arr3 := md.is_close(arr, arr2, 0.001, 0.001)
		defer md.free_mdarray(close_arr3)
		testing.expect_value(t, close_arr3.buffer[0], true)
		testing.expect_value(t, close_arr3.buffer[1], true)
	}

	{		
		arr := md.reshaped_range(f64, [2]int{2, 3})
		defer md.free_mdarray(arr)

		arr2 := md.reshaped_range(f64, [2]int{2, 3})
		defer md.free_mdarray(arr2)

		val := md.get_ref(arr, [2]int{0, 0})

		val^ += 0.00000001

		testing.expect_value(t, md.all_close(arr, arr2), true)
		
		val^ += 0.0001

		testing.expect_value(t, md.all_close(arr, arr2), false)
		testing.expect_value(t, md.all_close(arr, arr2, 0.001, 0.001), true)
	}
}
