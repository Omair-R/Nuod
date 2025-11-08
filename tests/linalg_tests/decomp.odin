package linalg_tests

import "core:log"
import "core:math"
import "core:testing"

@require import md "../../nuod/mdarray"
@require import nl "../../nuod/linalg"


//TODO: test for orthogonality and upper triangularity
@test
test_lapack_qr :: proc (t : ^testing.T){

	m:=3
	n:=2

	{
		arr := md.reshaped_range(f64, [2]int{m, n}, 1)
	
		qrr, rrr, ok := nl.qr(arr)
	
		arr_ := nl.matmul(qrr, rrr)

		testing.expect(t, md.all_close(arr, arr_))

		md.free_mdarray(arr)
		md.free_mdarray(arr_)
		md.free_mdarray(rrr)
		md.free_mdarray(qrr)
	
	}

	{
		arr := md.reshaped_range(f32, [2]int{n, m}, 1)
	
		qrr, rrr, ok := nl.qr(arr)
	
		arr_ := nl.matmul(qrr, rrr)

		testing.expect(t, md.all_close(arr, arr_))

		md.free_mdarray(arr)
		md.free_mdarray(arr_)
		md.free_mdarray(rrr)
		md.free_mdarray(qrr)
	
	}

	{
		arr := md.reshaped_range(f64, [3]int{3, m, n}, 1)

		qrr, rrr, ok := nl.qr(arr)

		arr_ := nl.matmul(qrr, rrr)

		testing.expect(t, md.all_close(arr, arr_))

		md.free_mdarray(arr)
		md.free_mdarray(arr_)
		md.free_mdarray(rrr)
		md.free_mdarray(qrr)
	}
}
