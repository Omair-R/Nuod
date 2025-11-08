package linalg


import lapacke "../lapacke"
import logging "../logging"
import "core:c"
import "base:intrinsics"


lapack_qr_wrapper :: proc(
	a: []$T,
	m, n: lapacke.blasint,
	r_mat : []T,
	q_mat : []T,
	allocator:=context.allocator,
	location:= #caller_location,
)->(
	ok: bool
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {

	if T != f32 && T != f64 && T != complex64 && T != complex128 {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 types.",
			location = location,
		)
		return
	}

	when ODIN_DEBUG {
		if m <= 0 || n <= 0 {
			logging.error(
				.ArguementError,
				"m, n cannot be set to <= 0.",
				location = location,
			)
			return 
		}
	}

	k := min(m, n)
	lda := n
	
	tau, err := make([]T, k, allocator)
	if err != .None {		
		logging.error(
			.AllocationError,
			location = location,
		)
		return
	}
	defer delete(tau)
	
	info : lapacke.blasint

	when T == f32{
		info = lapacke.sgeqrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(tau),
		)
	} else when T == f64{
		info = lapacke.dgeqrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(tau),
		)
	} else when T == complex64{
		info = lapacke.cgeqrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(tau),
		)
	} else when T == complex128{
		info = lapacke.zgeqrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(tau),
		)
	}


	if info != 0 {
		logging.error(
			.ArithmeticError,
			"An error occured during the performance of the QR subroutines.",
			location = location,
		)
		return
	}

	
	for i in 0..<k{
		begin:= i*n+i
		end:= n*(i+1)
		copy(r_mat[begin:end], a[begin:end])
	}

	vect: c.char
	if m>=n do vect = 'Q'
	else do vect = 'P'
	
	when T == f32{
		info = lapacke.sorgqr(
			lapacke.LAPACK_ROW_MAJOR,
			m, k, k,
			raw_data(a), lda,
			raw_data(tau),
		)
	} else when T == f64{
		info = lapacke.dorgqr(
			lapacke.LAPACK_ROW_MAJOR,
			m, k, k,
			raw_data(a), lda,
			raw_data(tau),
		)
	} else when T == complex64{
		info = lapacke.cungbr(
			lapacke.LAPACK_ROW_MAJOR,
			vect,
			m, k, k,
			raw_data(a), lda,
			raw_data(tau),
		)
	} else when T == complex128{
		info = lapacke.zungbr(
			lapacke.LAPACK_ROW_MAJOR,
			vect,
			m, k, k,
			raw_data(a), lda,
			raw_data(tau),
		)
	}

	if info != 0 {
		logging.error(
			.ArithmeticError,
			"An error occured during the performance of the QR subroutines.",
			location = location,
		)
		delete(q_mat)
		return
	}

	
	for i in 0..<m{
		end_q:= i*k+k
		end_a:= i*n+k
		copy(q_mat[i*k:end_q], a[i*n:end_a])
	}

	return true
}
