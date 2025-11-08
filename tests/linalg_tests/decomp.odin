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
	

		s_, ok2 := nl.svd_skip_uv(2, arr)

		
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

	
	// mode := nl.SVD_Mode.Full

	// m:=4
	// n:=2
	// k := int(min(m, n))

	// u_size, v_size : int
	// switch mode {
	// 	case .Full:
	// 		u_size = int(m*m)
	// 		v_size = int(n*n)
	// 	case .Reduced:
	// 		u_size = int(m*k)
	// 		v_size = int(n*k)
	// 	case .Skip_UV:
	// 		u_size = 0
	// 		v_size = 0
	// }

	// a := make([]f64, m*n)
	// s := make([]f64, k)

	// for i in 1..<m*n+1{
	// 	a[i-1] = f64(i)
	// }
	// defer delete(a)
	// defer delete(s)	

	// u : []f64
	// vt : []f64
	// if mode != .Skip_UV{
	// 	u = make([]f64, u_size)
	// 	vt = make([]f64, v_size)
	// }
	// defer if mode != .Skip_UV{
	// 	delete(u)
	// 	delete(vt)
	// }

	// nl.lapack_svd_wrapper(
	// 	a, i64(m), i64(n),
	// 	s, u, vt, mode,  
	// )

	// log.info(s)
	// log.info(u)
	// log.info(vt)
	
}
