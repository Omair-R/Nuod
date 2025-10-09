package linalg
import cblas "../cblas"
import "base:intrinsics"


cblas_real_norm2_wrapper :: proc (x: []$T) -> (
	result: T,
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok
{

	when T == f32{
		result = cblas.snrm2(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else when T == f64{
		result = cblas.dnrm2(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 types.",
			location = location,
		)
	}

	return result, true
}


cblas_cmplx_norm2_wrapper :: proc (x: []$T, $R: typeid) -> (
	result: R,
	ok: bool,
) where intrinsics.type_is_complex(T) #optional_ok
{
	when T == complex64{
		result = cblas.scnrm2(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else when T == complex128{
		result = cblas.dznrm2(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support complex64 and complex128 types.",
			location = location,
		)
	}

	return result, true
}


cblas_real_norm1_wrapper :: proc (x: []$T) -> (
	result: T,
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok
{

	when T == f32{
		result = cblas.sasum(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else when T == f64{
		result = cblas.dasum(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 types.",
			location = location,
		)
	}

	return result, true
}


cblas_cmplx_norm1_wrapper :: proc (x: []$T, $R: typeid) -> (
	result: R,
	ok: bool,
) where intrinsics.type_is_complex(T) #optional_ok
{
	when T == complex64{
		result = cblas.scasum(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else when T == complex128{
		result = cblas.dzasum(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support complex64 and complex128 types.",
			location = location,
		)
	}

	return result, true
}


cblas_real_norminfty_wrapper :: proc (x: []$T) -> (
	result: T,
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok
{

	when T == f32{
		result = cblas.sasum(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else when T == f64{
		result = cblas.dasum(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 types.",
			location = location,
		)
	}

	return result, true
}


cblas_cmplx_norminfty_wrapper :: proc (x: []$T, $R: typeid) -> (
	result: R,
	ok: bool,
) where intrinsics.type_is_complex(T) #optional_ok
{
	when T == complex64{
		result = cblas.scamax(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else when T == complex128{
		result = cblas.dzamax(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support complex64 and complex128 types.",
			location = location,
		)
	}

	return result, true
}


cblas_dot_wrapper :: proc (x: []$T, y: []T) -> (
	result: T,
	ok: bool,
) where intrinsics.type_is_float(T)|| intrinsics.type_is_complex(T) #optional_ok
{
	when ODIN_DEBUG {
		if len(x) != len(y) {
			logging.error(
				.ArguementError,
				"Vectors sizes are miss matched.",
				location = location,
			)
			return 
		}
	}

	when T == f32{
		result = cblas.sdot(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
			raw_data(y),
			cblas.blasint(1),
		)
	} else when T == f64{
		result = cblas.ddot(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
			raw_data(y),
			cblas.blasint(1),
		)
	} else when T == complex64 {
		result = cblas.cdotu(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
			raw_data(y),
			cblas.blasint(1),
		)
	} else when T == complex128 {
		result = cblas.zdotu(
			cblas.blasint(len(x)),
			raw_data(x),
			cblas.blasint(1),
			raw_data(y),
			cblas.blasint(1),
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 types.",
			location = location,
		)
	}

	return result, true
}
	

cblas_matvec_wrapper :: proc (
	a:[]$T, v:[]T,
	m, n: cblas.blasint,
	w_out: []T,
	transpose_a:= false,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T){
		
	when ODIN_DEBUG {
		if m <= 0 || n <= 0 {
			logging.error(
				.ArguementError,
				"m, n cannot be set to <= 0.",
				location = location,
			)
			return 
		}
		if !transpose_a && (len(v) != n || len(w_out) != m){
			logging.error(
				.ArguementError,
				"Inconsistance shape with the length of provided arrays.",
				location = location,
			)
			return 
		}
		if transpose_a && (len(v) != m || len(w_out) != n){
			logging.error(
				.ArguementError,
				"Inconsistance shape with the length of provided arrays.",
				location = location,
			)
			return 
		}
	}

	
	trans_a : cblas.CBLAS_TRANSPOSE = transpose_a? .Trans : .NoTrans

	lda: cblas.blasint = n


	when T == f32 {
		cblas.sgemv(
			.RowMajor,
			trans_a,
			m, n,
			1.0,
			raw_data(a),
			lda,
			raw_data(v),
			cblas.blasint(1),
			0.0,
			raw_data(w_out),
			cblas.blasint(1),
		)
	} else when T == f64{
		cblas.dgemv(
			.RowMajor,
			trans_a,
			m, n,
			1.0,
			raw_data(a),
			lda,
			raw_data(v),
			cblas.blasint(1),
			0.0,
			raw_data(w_out),
			cblas.blasint(1),
		)
	} else when T == complex64 {
		alpha :complex64 = complex(1.0, 0.0)
		beta :complex64 = complex(0.0, 0.0)
		cblas.cgemv(
			.RowMajor,
			trans_a,
			m, n, 
			&alpha,
			raw_data(a),
			lda,
			raw_data(v),
			cblas.blasint(1),
			&beta,
			raw_data(w_out),
			cblas.blasint(1)
		)
	} else when T == complex128 {
		alpha :complex128 = complex(1.0, 0.0)
		beta :complex128 = complex(0.0, 0.0)
		cblas.zgemv(
			.RowMajor,
			trans_a,
			m, n, 
			&alpha,
			raw_data(a),
			lda,
			raw_data(v),
			cblas.blasint(1),
			&beta,
			raw_data(w_out),
			cblas.blasint(1)
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 type.",
			location = location,
		)
		return
	}

	return true
}


cblas_matmul_wrapper :: proc (
	a: []$T, b: []T,
	m, n, k: cblas.blasint,
	c_out: []T,
	transpose_a:=false,
	transpose_b:=false
) -> (
	ok: bool
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) 
{
	when ODIN_DEBUG {
		if m <= 0 || n <= 0 || k <= 0 {
			logging.error(
				.ArguementError,
				"m, n, k cannot be set to <= 0.",
				location = location,
			)
			return 
		}
		if len(a) != (m * k) || len(b) != (n * k) || len(c_out) != (m * n){
			logging.error(
				.ArguementError,
				"Inconsistance shape with the length of provided arrays.",
				location = location,
			)
			return 
		}
	}

	trans_a : cblas.CBLAS_TRANSPOSE = .NoTrans
	trans_b : cblas.CBLAS_TRANSPOSE = .NoTrans

	lda: cblas.blasint = k
	ldb: cblas.blasint = n
	ldc: cblas.blasint = n

	if transpose_a { // m is the number of rows after transposing in this case
		trans_a = .Trans
		lda = m
	}

	if transpose_b { // same case for n here. 
		trans_b = .Trans
		ldb = k
	}

	when T == f32 {
		cblas.sgemm(
			.RowMajor,
			trans_a,
			trans_b,
			m, n, k,
			1.0,
			raw_data(a), lda,
			raw_data(b), ldb, 0.0,
			raw_data(c_out), ldc
		)
	} else when T == f64{
		cblas.dgemm(
			.RowMajor,
			trans_a,
			trans_b,
			m, n, k,
			1.0,
			raw_data(a), lda,
			raw_data(b), ldb, 0.0,
			raw_data(c_out), ldc
		)
	} else when T == complex64 {
		alpha :complex64 = complex(1.0, 0.0)
		beta :complex64 = complex(0.0, 0.0)
		cblas.cgemm(
			.RowMajor,
			trans_a,
			trans_b,
			m, n, k,
			&alpha,
			raw_data(a), lda,
			raw_data(b), ldb,
			&beta,
			raw_data(c_out), ldc
		)
	} else when T == complex128 {
		alpha :complex128 = complex(1.0, 0.0)
		beta :complex128 = complex(0.0, 0.0)
		cblas.zgemm(
			.RowMajor,
			trans_a,
			trans_b,
			m, n, k,
			&alpha,
			raw_data(a), lda,
			raw_data(b), ldb,
			&beta,
			raw_data(c_out), ldc
		)
	} else {
		logging.error(
			.ArguementError,
			"OpenBlas functions only support f32 and f64 types.",
			location = location,
		)
		return
	}

	return true
}

