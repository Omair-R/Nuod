package linalg
import cblas "../cblas"
import "base:intrinsics"



cblas_matmul_wrapper :: proc (a: []$T, b: []T, m, n, k: cblas.blasint, c_out: []T, transpose_a:=false, transpose_b:=false) -> (
	ok: bool
) where intrinsics.type_is_float(T) 
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
		if len(a) != (m * k) || len(b) != (n * k) || len(c) != (m * n){
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
	} else {
		if len(a) != (m * k) || len(b) != (n * k) || len(c) != (m * n){
			logging.error(
				.ArguementError,
				"OpenBlas only supports f32 and f64.",
				location = location,
			)
			return 
		}
		return
	}

	return true
}

