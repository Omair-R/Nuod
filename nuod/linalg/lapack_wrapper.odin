package linalg


import lapacke "../lapacke"
import logging "../logging"
import "core:c"
import "base:intrinsics"


SVD_Mode :: enum{
	Full,
	Reduced,
	Skip_UV
}

@private
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


@private
lapack_svd_wrapper :: proc(
	a: []$T,
	m, n: lapacke.blasint,
	s : []T,
	u : []T,
	vt : []T,
	mode: SVD_Mode,
	location:= #caller_location,
)->(
	ok: bool
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {

	k := int(min(m, n))

	u_size, v_size : int
	switch mode {
		case .Full:
			u_size = int(m*m)
			v_size = int(n*n)
		case .Reduced:
			u_size = int(m)*k
			v_size = int(n)*k
		case .Skip_UV:
			u_size = 0
			v_size = 0
	}

	
	when ODIN_DEBUG {
		if T != f32 && T != f64 && T != complex64 && T != complex128 {
			logging.error(
				.ArguementError,
				"OpenBlas functions only support f32 and f64 types.",
				location = location,
			)
			return
		}

		if len(s) != k || len(u) != u_size || len(vt) != v_size {
			logging.error(
				.ArguementError,
				"Incorrect output array size for s, u or vt.",
				location = location,
			)
			return
		}

		if m <= 0 || n <= 0 {
			logging.error(
				.ArguementError,
				"m, n cannot be set to <= 0.",
				location = location,
			)
			return 
		}
	}

	lda := n
	//the leading dimension for ldu depends on k, when in reduced mode
	ldu := m>n && mode == .Reduced? n : m
	ldv := n

	info : lapacke.blasint
	
	s_mode : c.char
	switch mode {
		case .Full:
			s_mode = 'A'
		case .Reduced:
			s_mode = 'S'
		case .Skip_UV:
			s_mode = 'N'
	}

	when T == f32{
		info = lapacke.sgesdd(
			lapacke.LAPACK_ROW_MAJOR,
			s_mode, m, n,
			raw_data(a), lda,
			raw_data(s),
			raw_data(u), ldu,
			raw_data(vt), ldv
		)
	} else when T == f64{
		info = lapacke.dgesdd(
			lapacke.LAPACK_ROW_MAJOR,
			s_mode, m, n,
			raw_data(a), lda,
			raw_data(s),
			raw_data(u), ldu,
			raw_data(vt), ldv
		)
	} else when T == complex64{
		info = lapacke.cgesdd(
			lapacke.LAPACK_ROW_MAJOR,
			s_mode, m, n,
			raw_data(a), lda,
			raw_data(s),
			raw_data(u), ldu,
			raw_data(vt), ldv
		)
	} else when T == complex128{
		info = lapacke.zgesdd(
			lapacke.LAPACK_ROW_MAJOR,
			s_mode, m, n,
			raw_data(a), lda,
			raw_data(s),
			raw_data(u), ldu,
			raw_data(vt), ldv
		)
	}

	if info > 0 {
		logging.error(
			.ArithmeticError,
			"the SVD subroutines failed to converge.",
			location = location,
		)
		return
	}

	return true
}


lapack_eig_wrapper_f :: proc(
	a: []$T,
	n: lapacke.blasint,
	wr: []T,
	wi: []T,
	vr : []T,
	compute_vecs: bool,
	location:= #caller_location,
)->(
	ok: bool
) where intrinsics.type_is_float(T){

	when ODIN_DEBUG {
		if T != f32 && T != f64 {
			logging.error(
				.ArguementError,
				"OpenBlas functions only support f32 and f64 types.",
				location = location,
			)
			return
		}
		if m <= 0 || n <= 0 {
			logging.error(
				.ArguementError,
				"m, n cannot be set to <= 0.",
				location = location,
			)
			return 
		}
		if len(e_vals) != n || len(e_vecs) != n*n {
			logging.error(
				.ArguementError,
				"Incorrect output array size for s, u or vt.",
				location = location,
			)
			return
		}
	}

	lda := n
	ldvr := n
	ldvl := n

	info : lapacke.blasint

	jobvr : c.char = compute_vecs? 'V' : 'N'

	when T == f32{
		info = lapacke.sgeev(
			lapacke.LAPACK_ROW_MAJOR,
			'N', jobvr, n,
			raw_data(a), lda,
			raw_data(wr), raw_data(wi),
			raw_data([]T{}), ldvl,
			raw_data(vr), ldvr
		)
	} else when T == f64{
		info = lapacke.dgeev(
			lapacke.LAPACK_ROW_MAJOR,
			'N', jobvr, n,
			raw_data(a), lda,
			raw_data(wr), raw_data(wi),
			raw_data([]T{}), ldvl,
			raw_data(vr), ldvr
		)
	}

	if info > 0 {
		logging.error(
			.ArithmeticError,
			"the eig subroutines failed to converge.",
			location = location,
		)
		return
	}

	return true
}


@private
lapack_lu_wrapper :: proc(
	a: []$T,
	m, n: lapacke.blasint,
	ipiv : []lapacke.blasint,
	location:= #caller_location,
)->(
	ok: bool
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {

	k := int(min(m, n))
	
	when ODIN_DEBUG {
		if T != f32 && T != f64 && T != complex64 && T != complex128 {
			logging.error(
				.ArguementError,
				"OpenBlas functions only support f32 and f64 types.",
				location = location,
			)
			return
		}

		if len(a) != m*n || len(ipiv) != k {
			logging.error(
				.ArguementError,
				"Incorrect output array size for s, u or vt.",
				location = location,
			)
			return
		}

		if m <= 0 || n <= 0 {
			logging.error(
				.ArguementError,
				"m, n cannot be set to <= 0.",
				location = location,
			)
			return 
		}
	}

	lda := n

	info : lapacke.blasint
	
	when T == f32{
		info = lapacke.sgetrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	} else when T == f64{
		info = lapacke.dgetrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	} else when T == complex64{
		info = lapacke.cgetrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	} else when T == complex128{
		info = lapacke.zgetrf(
			lapacke.LAPACK_ROW_MAJOR,
			m, n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	}

	if info > 0 {
		logging.warning(
			.None, // TODO
			"the diagonal of the U matrix is exactly zero, do no use to solve a system of equation.",
			location = location,
		)
		return false
	}

	return true
}


@private
lapack_lu_inv_wrapper :: proc(
	a: []$T,
	n: lapacke.blasint,
	ipiv : []lapacke.blasint,
	location:= #caller_location,
)->(
	ok: bool
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {

	lda := n

	lapack_lu_wrapper(a, n, n, ipiv, location) or_return	

	info : lapacke.blasint
	when T == f32{
		info = lapacke.sgetri(
			lapacke.LAPACK_ROW_MAJOR,
			n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	} else when T == f64{
		info = lapacke.dgetri(
			lapacke.LAPACK_ROW_MAJOR,
			n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	} else when T == complex64{
		info = lapacke.cgetri(
			lapacke.LAPACK_ROW_MAJOR,
			n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	} else when T == complex128{
		info = lapacke.zgetri(
			lapacke.LAPACK_ROW_MAJOR,
			n,
			raw_data(a), lda,
			raw_data(ipiv)
		)
	}

	if info > 0 {
		logging.error(
			.ArithmeticError,
			"Matrix is singular, there is no inverse.",
			location= location,
		)
		return
	}

	return true
}


@private
lapack_solve_wrapper :: proc(
	a: []$T,
	b: []T,
	n, nrhs: lapacke.blasint,
	ipiv : []lapacke.blasint,
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
		if nrhs <= 0 || n <= 0 {
			logging.error(
				.ArguementError,
				"m, n cannot be set to <= 0.",
				location = location,
			)
			return 
		}
	}

	lda := n
	ldb := nrhs
	
	info : lapacke.blasint

	when T == f32{
		info = lapacke.sgesv(
			lapacke.LAPACK_ROW_MAJOR,
			n, nrhs,
			raw_data(a), lda,
			raw_data(ipiv),
			raw_data(b), ldb
		)
	} else when T == f64{
		info = lapacke.dgesv(
			lapacke.LAPACK_ROW_MAJOR,
			n, nrhs,
			raw_data(a), lda,
			raw_data(ipiv),
			raw_data(b), ldb
		)
	} else when T == complex64{
		info = lapacke.cgesv(
			lapacke.LAPACK_ROW_MAJOR,
			n, nrhs,
			raw_data(a), lda,
			raw_data(ipiv),
			raw_data(b), ldb
		)
	} else when T == complex128{
		info = lapacke.zgesv(
			lapacke.LAPACK_ROW_MAJOR,
			n, nrhs,
			raw_data(a), lda,
			raw_data(ipiv),
			raw_data(b), ldb
		)
	}

	if info != 0 {
		logging.error(
			.ArithmeticError,
			"Matrix A is singular, resulting in a division by 0.",
			location = location,
		)
		return
	}

	return true
}


