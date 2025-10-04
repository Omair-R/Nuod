package mdarray_tests

import "core:log"
import "core:testing"

@require import md "../../nuod/mdarray"



@private
make_sq_f :: proc($T: typeid)-> (proc(^T)) {
	return proc(val: ^T)  { val^ = val^ * val^ }
}


@(test)
test_inplace_unary :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f64, [2]int{3, 3})
		defer md.free_mdarray(arr)

		ok := md.inplace_unary_map(arr, make_sq_f(f64))
		testing.expect(t, ok)
		testing.expect_value(t, arr.shape, [2]int{3, 3})

		expect_arr := []f64{0, 1, 4, 9, 16, 25, 36, 49, 64}
		for i in 0..<9{
			testing.expect_value(t, arr.buffer[i], expect_arr[i])
		}
	}

	{
		arr := md.reshaped_range(f64, [2]int{3, 3})
		defer md.free_mdarray(arr)

		ok := md.inplace_sq(arr)
		testing.expect(t, ok)
		testing.expect_value(t, arr.shape, [2]int{3, 3})

		expect_arr := []f64{0, 1, 4, 9, 16, 25, 36, 49, 64}
		for i in 0..<9{
			testing.expect_value(t, arr.buffer[i], expect_arr[i])
		}
		
		ok = md.inplace_neg(arr)
		testing.expect(t, ok)
		for i in 0..<9{
			testing.expect_value(t, arr.buffer[i], -expect_arr[i])
		}

		ok = md.inplace_neg(arr)
		testing.expect(t, ok)
		ok = md.inplace_sqrt(arr)
		testing.expect(t, ok)

		expect_arr = []f64{0, 1, 2, 3, 4, 5, 6, 7, 8}
		for i in 0..<9{
			testing.expect_value(t, arr.buffer[i], expect_arr[i])
		}
	}
}


@(test)
test_outplace_unary :: proc(t: ^testing.T){
	{
		arr := md.reshaped_range(f64, [2]int{3, 3})
		defer md.free_mdarray(arr)

		sq_arr := md.outplace_unary_map(arr, make_sq_f(f64))
		defer md.free_mdarray(sq_arr)

		testing.expect_value(t, sq_arr.shape, [2]int{3, 3})
		testing.expect_value(t, len(sq_arr.buffer), 9)

		expect_arr := []f64{0, 1, 4, 9, 16, 25, 36, 49, 64}
		for i in 0..<9{
			testing.expect_value(t, sq_arr.buffer[i], expect_arr[i])
		}
	}

	{
		arr := md.reshaped_range(f64, [2]int{3, 3})
		defer md.free_mdarray(arr)

		sq_arr := md.outplace_sq(arr)
		defer md.free_mdarray(sq_arr)

		testing.expect_value(t, sq_arr.shape, [2]int{3, 3})

		expect_arr := []f64{0, 1, 4, 9, 16, 25, 36, 49, 64}
		for i in 0..<9{
			testing.expect_value(t, sq_arr.buffer[i], expect_arr[i])
		}
		
		neg_arr := md.outplace_neg(sq_arr)
		defer md.free_mdarray(neg_arr)

		for i in 0..<9{
			testing.expect_value(t, neg_arr.buffer[i], -expect_arr[i])
		}

		abs_arr := md.outplace_neg(neg_arr)
		defer md.free_mdarray(abs_arr)

		for i in 0..<9{
			testing.expect_value(t, abs_arr.buffer[i], expect_arr[i])
		}

		sqrt_arr := md.outplace_sqrt(sq_arr)
		defer md.free_mdarray(sqrt_arr)

		expect_arr = []f64{0, 1, 2, 3, 4, 5, 6, 7, 8}
		for i in 0..<9{
			testing.expect_value(t, sqrt_arr.buffer[i], expect_arr[i])
		}
	}
}
