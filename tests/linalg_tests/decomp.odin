package linalg_tests

import "core:log"
import "core:math"
import "core:testing"

@require import md "../../nuod/mdarray"
@require import nl "../../nuod/linalg"
@require import nr "../../nuod/random"


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


// TODO: test for higher dimensions
@test
test_lapack_svd :: proc (t : ^testing.T){

	{
		m:=4
		n:=3

		arr := nr.normal_sample(f64(0), f64(1), [2]int{m, n})
	
		s, u, vt, ok := nl.reduced_svd(2, arr)
	
		
		s_diag := nl.make_diagonal(s)
		u_s := nl.matmul(u, s_diag)
		arr_ := nl.matmul(u_s, vt)

		testing.expect(t, md.all_close(arr, arr_))

		md.free_mdarray(arr)
		md.free_mdarray(s)
		md.free_mdarray(s_diag)
		md.free_mdarray(u)
		md.free_mdarray(vt)
	
		md.free_mdarray(u_s)
		md.free_mdarray(arr_)
	}

	{
		m:=4
		n:=3

		arr := nr.normal_sample(f64(0), f64(1), [2]int{m, n})
	
		s, u, vt, ok := nl.full_svd(2, arr)
	

		s_, ok2 := nl.svd_vals(2, arr)

		
		testing.expect(t, md.all_close(s, s_))
		testing.expect_value(t, s.shape, [1]int{3})
		testing.expect_value(t, u.shape, [2]int{4, 4})
		testing.expect_value(t, vt.shape, [2]int{3, 3})

		md.free_mdarray(arr)
		md.free_mdarray(s)
		md.free_mdarray(s_)
		md.free_mdarray(u)
		md.free_mdarray(vt)
	}
}


// TODO: Test that decomposition is equal and test higher dimensions
@test
test_lapack_eig :: proc (t : ^testing.T){

	{
		n:= 2

		arr := md.reshaped_range(f64, [2]int{n, n}, 1)

		e_vals, e_vecs, ok := nl.eig(2, arr)

		testing.expect_value(t, e_vals.shape, [1]int{n})
		testing.expect_value(t, e_vecs.shape, [2]int{n, n})
		
		md.free_mdarray(arr)
		md.free_mdarray(e_vals)
		md.free_mdarray(e_vecs)
	}
	
}
