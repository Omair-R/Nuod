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

	{
		arr := md.reshaped_range(complex64, [2]int{m, n}, 1)
	
		qrr, rrr, ok := nl.qr(arr)
	
		arr_ := nl.matmul(qrr, rrr)

		testing.expect(t, md.all_close(arr, arr_))

		md.free_mdarray(arr)
		md.free_mdarray(arr_)
		md.free_mdarray(rrr)
		md.free_mdarray(qrr)
	
	}
	{
		arr := md.reshaped_range(complex128, [3]int{3, m, n}, 1)

		qrr, rrr, ok := nl.qr(arr)

		arr_ := nl.matmul(qrr, rrr)

		testing.expect(t, md.all_close(arr, arr_))

		md.free_mdarray(arr)
		md.free_mdarray(arr_)
		md.free_mdarray(rrr)
		md.free_mdarray(qrr)
	}
}


@test
test_lapack_svd :: proc (t : ^testing.T){

	{
		m:=4
		n:=3

		arr := nr.normal_sample(f32(0), f32(1), [2]int{m, n})
		s, u, vt, ok := nl.reduced_svd(2, arr)

		s_ := nl.svd_vals(2, arr)
		defer md.free_mdarray(s_)
		testing.expect(t, md.all_close(s, s_))
		
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
	
		s, u, vt, ok := nl.reduced_svd(2, arr)
	
		s_ := nl.svd_vals(2, arr)
		defer md.free_mdarray(s_)
		testing.expect(t, md.all_close(s, s_))
		
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

		arr := md.reshaped_range(complex64, [2]int{m, n}, 1)
	
		s, u, vt, ok := nl.reduced_svd(2, arr)
	
		s_ := nl.svd_vals(2, arr)
		defer md.free_mdarray(s_)
		testing.expect(t, md.all_close(s, s_))
		
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

		arr := md.reshaped_range(complex128, [2]int{m, n}, 1)
	
		s, u, vt, ok := nl.reduced_svd(2, arr)
	
		s_ := nl.svd_vals(2, arr)
		defer md.free_mdarray(s_)
		testing.expect(t, md.all_close(s, s_))
		
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

		arr := nr.normal_sample(f64(0), f64(1), [3]int{2, m, n})
	
		s, u, vt, ok := nl.reduced_svd(3, arr)
		
		s_diag := nl.make_diagonal(2, s)
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

		arr := nr.normal_sample(f64(0), f64(1), [4]int{2, 2, m, n})
	
		s, u, vt, ok := nl.reduced_svd(4, arr)
		
		s_diag := nl.make_diagonal(3, s)
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


@test
test_lapack_eig :: proc (t : ^testing.T){

	{
		n:= 2

		arr := md.reshaped_range(f64, [2]int{n, n}, 1)

		e_vals, e_vecs, ok := nl.eig(2, arr)

		testing.expect(t, ok)
		testing.expect_value(t, e_vals.shape, [1]int{n})
		testing.expect_value(t, e_vecs.shape, [2]int{n, n})

		
		ev_diag := nl.make_diagonal(e_vals)
		evc_inv := nl.inv(e_vecs)
		intr := nl.matmul(e_vecs, ev_diag)
		arr_ := nl.matmul(intr, evc_inv)
		arr_c := md.cast_array(arr, complex128)
		testing.expect(t, md.all_close(arr_c, arr_))
		
		md.free_mdarray(arr)
		md.free_mdarray(e_vals)
		md.free_mdarray(e_vecs)

		md.free_mdarray(ev_diag)
		md.free_mdarray(evc_inv)
		md.free_mdarray(intr)
		md.free_mdarray(arr_)
		md.free_mdarray(arr_c)
	}

	
	{
		n:= 2

		arr := md.reshaped_range(f64, [3]int{2, n, n}, 1)

		e_vals, e_vecs, ok := nl.eig(3, arr)

		testing.expect(t, ok)
		testing.expect_value(t, e_vals.shape, [2]int{2, n})
		testing.expect_value(t, e_vecs.shape, [3]int{2, n, n})
		
		md.free_mdarray(arr)
		md.free_mdarray(e_vals)
		md.free_mdarray(e_vecs)
	}
	
}

@test
test_lapack_eigvals :: proc (t : ^testing.T){

	{
		n:= 2

		arr := md.reshaped_range(f64, [2]int{n, n}, 1)

		e_vals, ok := nl.eigvals(2, arr)

		testing.expect(t, ok)

		e_vals2, e_vecs, ok2 := nl.eig(2, arr)

		testing.expect(t, ok2)
		testing.expect_value(t, e_vals.shape, [1]int{n})
		
		md.free_mdarray(arr)
		md.free_mdarray(e_vals)
		md.free_mdarray(e_vals2)
		md.free_mdarray(e_vecs)
	}

	{
		n:= 2

		arr := md.reshaped_range(f64, [3]int{2, n, n}, 1)

		e_vals, ok := nl.eigvals(3, arr)

		testing.expect(t, ok)

		e_vals2, e_vecs, ok2 := nl.eig(3, arr)

		testing.expect(t, ok2)
		testing.expect_value(t, e_vals.shape, [2]int{2, n})
		
		md.free_mdarray(arr)
		md.free_mdarray(e_vals)
		md.free_mdarray(e_vals2)
		md.free_mdarray(e_vecs)
	}
	
}


@test
test_lapack_det :: proc (t : ^testing.T){

	{
		n:= 3

		arr := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})

		det, ok := nl.det(arr)

		testing.expect_value(t, det, 8)
		md.free_mdarray(arr)
	}

	{
		n:= 3

		arr := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})
		arr_s := md.stack(2, []md.MdArray(f64, 2){arr, arr})
		det, ok := nl.det(3, arr_s)

		testing.expect_value(t, det.buffer[0], 8)
		testing.expect_value(t, det.buffer[1], 8)
		md.free_mdarray(arr)
		md.free_mdarray(arr_s)
		md.free_mdarray(det)
	}

	{
		n:= 3

		arr := md.from_slice([]f64{1, 3, 0, 4, 1, 0, 2, 0, 1}, [2]int{n, n})

		det, ok := nl.det(arr)

		testing.expect_value(t, det, -11)
		md.free_mdarray(arr)
	}

	{
		n:= 3

		arr := md.from_slice([]f64{6, 1, 1, 4, -2, 5, 2, 8, 7}, [2]int{n, n})

		det, ok := nl.det(arr)

		testing.expect_value(t, det, -306)
		md.free_mdarray(arr)
	}

	{
		n:= 2

		arr := md.from_slice([]complex128{
			complex(-2, 0),
			complex(-9, 0),
			complex(1, 0),
			complex(4, 0)
		}, [2]int{n, n})

		det, ok := nl.det(arr)

		testing.expect_value(t, det, complex(1, 0))
		md.free_mdarray(arr)
	}

	{
		n:= 2

		arr := md.from_slice([]complex64{
			complex(-2, 0),
			complex(9, 0),
			complex(1, 0),
			complex(-4, 0)
		}, [2]int{n, n})

		det, ok := nl.det(arr)

		testing.expect_value(t, det, complex(-1, 0))
		md.free_mdarray(arr)
	}

	{
		n:= 2

		arr := md.from_slice([]f64{4, 6, 3, 8, 3, 8, 4, 6}, [3]int{2, n, n})

		det, ok := nl.det(3, arr)
		det_, ok2 := md.from_slice([]f64{14, -14}, [1]int{2})

		testing.expect(t, md.all_close(det, det_))
		md.free_mdarray(arr)
		md.free_mdarray(det)
		md.free_mdarray(det_)
	}

	{
		n:= 2

		arr := md.from_slice([]f64{2, 2, 2, 1}, [2]int{n, n})

		det, ok := nl.det(arr)

		testing.expect_value(t, det, -2)
		md.free_mdarray(arr)
	}
	
	{
		n:= 2

		arr := md.reshaped_range(f64, [2]int{n, n}, 1)

		det, ok := nl.det(arr)

		testing.expect_value(t, det, -2)
		md.free_mdarray(arr)
	}

	{
		n:= 3

		arr := md.reshaped_range(f64, [2]int{n, n}, 1)

		det, ok := nl.det(arr)

		testing.expect_value(t, det, 0)
		md.free_mdarray(arr)
	}
	{
		n:= 4

		arr := md.reshaped_range(f64, [2]int{n, n}, 1)

		det, ok := nl.det(arr)

		testing.expect_value(t, det, 0)
		md.free_mdarray(arr)
	}
}


@test
test_lapack_inv :: proc (t : ^testing.T){

	{
		n:= 3

		arr := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})

		inv := nl.inv(arr)

		id := nl.matmul(inv, arr)
		id_ := md.identity(f64, 3)

		close := md.all_close(id, id_)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

	{
		n:= 2

		arr := md.from_slice([]complex64{
			complex(-2, 0),
			complex(9, 0),
			complex(1, 0),
			complex(-4, 0)
		}, [2]int{n, n})

		inv := nl.inv(arr)

		id := nl.matmul(inv, arr)
		id_ := md.identity(complex64, 2)

		close := md.all_close(id, id_)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

	{
		n:= 2

		arr := md.from_slice([]complex128{
			complex(3, 0),
			complex(9, 0),
			complex(-8, 0),
			complex(-4, 0)
		}, [2]int{n, n})

		inv := nl.inv(arr)

		id := nl.matmul(inv, arr)
		id_ := md.identity(complex128, 2)

		close := md.all_close(id, id_)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

	{
		s:= 3
		n:: 2

		arr := md.reshaped_range(f64, [3]int{s, n, n})

		inv := nl.inv(arr)


		id := nl.matmul(inv, arr)
		id_ := md.identity(f64, n)

		for i in 0..<s{
			n_id := md.slice_view(3, id, i)
			close := md.all_close(id_, n_id)
			testing.expect(t, close )
		}

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

}

@test
test_lapack_pinv :: proc (t : ^testing.T){

	{
		n:= 3

		arr := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})

		pinv := nl.pinv(arr)
		inv := nl.inv(arr)

		id := nl.matmul(pinv, arr)
		id_ := md.identity(f64, 3)

		close := md.all_close(id, id_)
		testing.expect(t, close )

		close = md.all_close(inv, pinv)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(pinv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

	{

		arr := md.from_slice([]f64{3, 2,  2, 0, 4, 2}, [2]int{3, 2})

		pinv := nl.pinv(arr)

		id := nl.matmul(pinv, arr)
		id_ := md.identity(f64, 2)

		close := md.all_close(id, id_)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(pinv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

	{
		n:= 2

		arr := md.from_slice([]complex64{
			complex(-2, 0),
			complex(9, 0),
			complex(1, 0),
			complex(-4, 0)
		}, [2]int{n, n})

		inv := nl.pinv(arr)

		id := nl.matmul(inv, arr)
		id_ := md.identity(complex64, 2)

		close := md.all_close(id, id_)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

	{
		s:= 3
		n:: 2

		arr := md.reshaped_range(f64, [3]int{s, n, n})

		pinv := nl.inv(arr)
		inv := nl.inv(arr)

		id := nl.matmul(pinv, arr)
		id_ := md.identity(f64, n)

		for i in 0..<s{
			n_id := md.slice_view(3, id, i)
			close := md.all_close(id_, n_id)
			testing.expect(t, close )
		}

		close := md.all_close(inv, pinv)
		testing.expect(t, close )

		md.free_mdarray(arr)
		md.free_mdarray(inv)
		md.free_mdarray(pinv)
		md.free_mdarray(id)
		md.free_mdarray(id_)
	}

}



@test
test_lapack_solve :: proc (t : ^testing.T){

	{
		n:= 3

		a := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})
		b := md.from_slice([]f64{1, 3, 2}, [1]int{n})
		

		sol := nl.solve(a, b)
		sol_ := md.from_slice([]f64{1.25, -1.875, 0.25}, [1]int{n})

		close := md.all_close(sol, sol_)
		testing.expect(t, close)
	
		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
	}

	{
		n:= 3

		a := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})
		a_ := md.cast_array(a, complex64)
		b := md.from_slice([]f64{1, 3, 2}, [1]int{n})
		b_ := md.cast_array(b, complex64)

		sol := nl.solve(a_, b_)
		sol_ := md.from_slice([]f64{1.25, -1.875, 0.25}, [1]int{n})
		sol_c := md.cast_array(sol_, complex64)

		close := md.all_close(sol, sol_c)
		testing.expect(t, close)
	
		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(a_)
		md.free_mdarray(b_)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
		md.free_mdarray(sol_c)
	}

	{
		s:=2
		n:= 3

		a := md.from_slice([]f32{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})
		b := md.from_slice([]f32{1, 2, 3, 1, 2, 2}, [2]int{n, s})
		

		sol := nl.solve(a, b)
		sol_ := md.from_slice([]f32{1.25, 0.25, -1.875, 0.125, 0.25, 0.25}, [2]int{n, s})

		close := md.all_close(sol, sol_)
		testing.expect(t, close)
	
		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
	}
}


@test
test_lapack_lstsq :: proc (t : ^testing.T){

	{
		n:= 3

		a := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})
		b := md.from_slice([]f64{1, 3, 2}, [1]int{n})
		

		sol, res, ok := nl.lstsq(a, b)
		sol_ := md.from_slice([]f64{1.25, -1.875, 0.25}, [1]int{n})

		close := md.all_close(sol, sol_)
		testing.expect(t, close)
	
		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
	}

	{
		n:= 3

		a := md.from_slice([]f64{3, 2, 4, 2, 0, 2, 4, 2, 3}, [2]int{n, n})
		a_ := md.cast_array(a, complex128)
		b := md.from_slice([]f64{1, 3, 2}, [1]int{n})
		b_ := md.cast_array(b, complex128)

		sol, res, ok := nl.lstsq(a_, b_)
		sol_ := md.from_slice([]f64{1.25, -1.875, 0.25}, [1]int{n})
		sol_c := md.cast_array(sol_, complex128)

		close := md.all_close(sol, sol_c)
		testing.expect(t, close)
	
		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(a_)
		md.free_mdarray(b_)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
		md.free_mdarray(sol_c)
	}

	{
		m:= 3
		n:= 2

		a := md.from_slice([]f64{3, 2, 2, 0, 4, 2}, [2]int{m, n})
		b := md.from_slice([]f64{1, 3, 2}, [1]int{m})
		

		sol, res, ok := nl.lstsq(a, b)
		sol_ := md.from_slice([]f64{1.444444444444, -1.77777777777778}, [1]int{2})

		close := md.all_close(sol, sol_)
		testing.expect(t, close)

		res_ := md.fills(0.111111111111, [1]int{1})
		close = md.all_close(res, res_)
		testing.expect(t, close)

		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
		md.free_mdarray(res)
		md.free_mdarray(res_)
	}

	{
		m:= 3
		n:= 2

		a := md.from_slice([]f64{3, 2, 2, 0, 4, 2}, [2]int{m, n})
		b := md.from_slice([]f64{1, 1, 3, 3, 2, 2}, [2]int{m, 2})
		

		sol, res, ok := nl.lstsq(a, b)
		sol_ := md.from_slice([]f64{
			1.444444444444, 1.444444444444,
			-1.77777777777778, -1.77777777777778}, [2]int{n, n})

		close := md.all_close(sol, sol_)
		testing.expect(t, close)

		res_ := md.fills(0.111111111111, [1]int{2})
		close = md.all_close(res, res_)
		testing.expect(t, close)
		md.free_mdarray(a)
		md.free_mdarray(b)
		md.free_mdarray(sol)
		md.free_mdarray(sol_)
		md.free_mdarray(res)
		md.free_mdarray(res_)
	}
	
}
