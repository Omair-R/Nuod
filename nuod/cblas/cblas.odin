/*

	This file has been provided by kalsprite from the OpenBLAS bindings:
		https://github.com/kalsprite/odin-command

*/

package cblas

import "core:c"

_ :: c

when ODIN_OS == .Windows {
	foreign import openblas "../../vendors/openblas/lib/libopenblas.lib"
} else when ODIN_OS == .Linux {
	foreign import openblas "system:openblas"
} else when ODIN_OS == .Darwin {
	foreign import openblas "system:openblas"
}


/*Set the threading backend to a custom callback.*/
openblas_dojob_callback :: proc "c" (job_id: c.int, job_data: rawptr, thread_id: c.int)

openblas_threads_callback :: proc "c" (num_threads: c.int, dojob: openblas_dojob_callback, jobdata_elsize: c.int, jobdata_stride: c.size_t, jobdata: rawptr, num_jobs: c.int)

/* OpenBLAS is compiled for sequential use  */
SEQUENTIAL :: 0

/* OpenBLAS is compiled using normal threading model */
THREAD :: 1

/* OpenBLAS is compiled using OpenMP threading model */
OPENMP :: 2

// CONST ::

CBLAS_INDEX :: c.size_t

CBLAS_ORDER :: enum c.uint {
	RowMajor = 101,
	ColMajor = 102,
}

CBLAS_TRANSPOSE :: enum c.uint {
	NoTrans     = 111,
	Trans       = 112,
	ConjTrans   = 113,
	ConjNoTrans = 114,
}

CBLAS_UPLO :: enum c.uint {
	Upper = 121,
	Lower = 122,
}

CBLAS_DIAG :: enum c.uint {
	NonUnit = 131,
	Unit    = 132,
}

CBLAS_SIDE :: enum c.uint {
	Left  = 141,
	Right = 142,
}


@(default_calling_convention = "c", link_prefix = "")
foreign openblas {
	/*Set the number of threads on runtime.*/
	openblas_set_num_threads :: proc(num_threads: c.int) ---
	goto_set_num_threads :: proc(num_threads: c.int) ---
	openblas_set_num_threads_local :: proc(num_threads: c.int) -> c.int ---

	/*Get the number of threads on runtime.*/
	openblas_get_num_threads :: proc() -> c.int ---

	/*Get the number of physical processors (cores).*/
	openblas_get_num_procs :: proc() -> c.int ---

	/*Get the build configure on runtime.*/
	openblas_get_config :: proc() -> cstring ---

	/*Get the CPU corename on runtime.*/
	openblas_get_corename :: proc() -> cstring ---
	openblas_set_threads_callback_function :: proc(callback: openblas_threads_callback) ---

	/* Get the parallelization type which is used by OpenBLAS */
	openblas_get_parallel :: proc() -> c.int ---
}


@(default_calling_convention = "c", link_prefix = "cblas_")
foreign openblas {

	sdsdot :: proc(n: blasint, alpha: f32, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f32 ---
	dsdot :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f64 ---
	sdot :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f32 ---
	ddot :: proc(n: blasint, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) -> f64 ---
	cdotu :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) -> complex64 ---
	cdotc :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) -> complex64 ---
	zdotu :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) -> complex128 ---
	zdotc :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) -> complex128 ---
	cdotu_sub :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint, ret: ^complex64) ---
	cdotc_sub :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint, ret: ^complex64) ---
	zdotu_sub :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint, ret: ^complex128) ---
	zdotc_sub :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint, ret: ^complex128) ---

	sasum :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
	dasum :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
	scasum :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
	dzasum :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---
	ssum :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
	dsum :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
	scsum :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
	dzsum :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---

	snrm2 :: proc(N: blasint, X: [^]f32, incX: blasint) -> f32 ---
	dnrm2 :: proc(N: blasint, X: [^]f64, incX: blasint) -> f64 ---
	scnrm2 :: proc(N: blasint, X: [^]complex64, incX: blasint) -> f32 ---
	dznrm2 :: proc(N: blasint, X: [^]complex128, incX: blasint) -> f64 ---

	isamax :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
	idamax :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
	icamax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
	izamax :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---
	isamin :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
	idamin :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
	icamin :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
	izamin :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---
	samax :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
	damax :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
	scamax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
	dzamax :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---
	samin :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
	damin :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
	scamin :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
	dzamin :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---
	ismax :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
	idmax :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
	icmax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
	izmax :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---
	ismin :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
	idmin :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
	icmin :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
	izmin :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---

	saxpy :: proc(n: blasint, alpha: f32, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) ---
	daxpy :: proc(n: blasint, alpha: f64, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) ---
	caxpy :: proc(n: blasint, alpha: ^complex64, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
	zaxpy :: proc(n: blasint, alpha: ^complex128, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---
	caxpyc :: proc(n: blasint, alpha: ^complex64, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
	zaxpyc :: proc(n: blasint, alpha: ^complex128, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---

	scopy :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) ---
	dcopy :: proc(n: blasint, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) ---
	ccopy :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
	zcopy :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---

	sswap :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) ---
	dswap :: proc(n: blasint, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) ---
	cswap :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
	zswap :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---

	srot :: proc(N: blasint, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, _c: f32, s: f32) ---
	drot :: proc(N: blasint, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, _c: f64, s: f64) ---
	csrot :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incY: blasint, _c: f32, s: f32) ---
	zdrot :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incY: blasint, _c: f64, s: f64) ---
	srotg :: proc(a: ^f32, b: ^f32, _c: ^f32, s: ^f32) ---
	drotg :: proc(a: ^f64, b: ^f64, _c: ^f64, s: ^f64) ---
	crotg :: proc(a: ^complex64, b: ^complex64, _c: ^f32, s: ^complex64) ---
	zrotg :: proc(a: ^complex128, b: ^complex128, _c: ^f64, s: ^complex128) ---
	srotm :: proc(N: blasint, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, P: [^]f32) ---
	drotm :: proc(N: blasint, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, P: [^]f64) ---
	srotmg :: proc(d1: ^f32, d2: ^f32, b1: ^f32, b2: f32, P: [^]f32) ---
	drotmg :: proc(d1: ^f64, d2: ^f64, b1: ^f64, b2: f64, P: [^]f64) ---

	sscal :: proc(N: blasint, alpha: f32, X: [^]f32, incX: blasint) ---
	dscal :: proc(N: blasint, alpha: f64, X: [^]f64, incX: blasint) ---
	cscal :: proc(N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint) ---
	zscal :: proc(N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint) ---
	csscal :: proc(N: blasint, alpha: f32, X: [^]complex64, incX: blasint) ---
	zdscal :: proc(N: blasint, alpha: f64, X: [^]complex128, incX: blasint) ---

	sgemv :: proc(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: f32, a: [^]f32, lda: blasint, x: [^]f32, incx: blasint, beta: f32, y: [^]f32, incy: blasint) ---
	dgemv :: proc(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: f64, a: [^]f64, lda: blasint, x: [^]f64, incx: blasint, beta: f64, y: [^]f64, incy: blasint) ---
	cgemv :: proc(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: ^complex64, a: [^]complex64, lda: blasint, x: [^]complex64, incx: blasint, beta: ^complex64, y: [^]complex64, incy: blasint) ---
	zgemv :: proc(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: ^complex128, a: [^]complex128, lda: blasint, x: [^]complex128, incx: blasint, beta: ^complex128, y: [^]complex128, incy: blasint) ---

	sger :: proc(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: f32, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, A: [^]f32, lda: blasint) ---
	dger :: proc(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: f64, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, A: [^]f64, lda: blasint) ---
	cgeru :: proc(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, A: [^]complex64, lda: blasint) ---
	cgerc :: proc(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, A: [^]complex64, lda: blasint) ---
	zgeru :: proc(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, A: [^]complex128, lda: blasint) ---
	zgerc :: proc(order: CBLAS_ORDER, M: blasint, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, A: [^]complex128, lda: blasint) ---

	strsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint) ---
	dtrsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint) ---
	ctrsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint) ---
	ztrsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint) ---
	strmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint) ---
	dtrmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint) ---
	ctrmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint) ---
	ztrmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint) ---

	ssyr :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, X: [^]f32, incX: blasint, A: [^]f32, lda: blasint) ---
	dsyr :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, X: [^]f64, incX: blasint, A: [^]f64, lda: blasint) ---
	cher :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, X: [^]complex64, incX: blasint, A: [^]complex64, lda: blasint) ---
	zher :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, X: [^]complex128, incX: blasint, A: [^]complex128, lda: blasint) ---
	ssyr2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, A: [^]f32, lda: blasint) ---
	dsyr2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, A: [^]f64, lda: blasint) ---
	cher2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, A: [^]complex64, lda: blasint) ---
	zher2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, A: [^]complex128, lda: blasint) ---

	sgbmv :: proc(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: f32, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint, beta: f32, Y: [^]f32, incY: blasint) ---
	dgbmv :: proc(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: f64, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint, beta: f64, Y: [^]f64, incY: blasint) ---
	cgbmv :: proc(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint, beta: ^complex64, Y: [^]complex64, incY: blasint) ---
	zgbmv :: proc(order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, M: blasint, N: blasint, KL: blasint, KU: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint, beta: ^complex128, Y: [^]complex128, incY: blasint) ---

	ssbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint, beta: f32, Y: [^]f32, incY: blasint) ---
	dsbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint, beta: f64, Y: [^]f64, incY: blasint) ---

	stbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint) ---
	dtbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint) ---
	ctbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint) ---
	ztbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint) ---

	stbsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint) ---
	dtbsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint) ---
	ctbsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint) ---
	ztbsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, K: blasint, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint) ---

	stpmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]f32, X: [^]f32, incX: blasint) ---
	dtpmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]f64, X: [^]f64, incX: blasint) ---
	ctpmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]complex64, X: [^]complex64, incX: blasint) ---
	ztpmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]complex128, X: [^]complex128, incX: blasint) ---

	stpsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]f32, X: [^]f32, incX: blasint) ---
	dtpsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]f64, X: [^]f64, incX: blasint) ---
	ctpsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]complex64, X: [^]complex64, incX: blasint) ---
	ztpsv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, N: blasint, Ap: [^]complex128, X: [^]complex128, incX: blasint) ---

	ssymv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint, beta: f32, Y: [^]f32, incY: blasint) ---
	dsymv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint, beta: f64, Y: [^]f64, incY: blasint) ---

	chemv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint, beta: ^complex64, Y: [^]complex64, incY: blasint) ---
	zhemv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint, beta: ^complex128, Y: [^]complex128, incY: blasint) ---

	sspmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, Ap: [^]f32, X: [^]f32, incX: blasint, beta: f32, Y: [^]f32, incY: blasint) ---
	dspmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, Ap: [^]f64, X: [^]f64, incX: blasint, beta: f64, Y: [^]f64, incY: blasint) ---

	sspr :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, X: [^]f32, incX: blasint, Ap: [^]f32) ---
	dspr :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, X: [^]f64, incX: blasint, Ap: [^]f64) ---
	chpr :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, X: [^]complex64, incX: blasint, A: [^]complex64) ---
	zhpr :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, X: [^]complex128, incX: blasint, A: [^]complex128) ---
	sspr2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f32, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, A: [^]f32) ---
	dspr2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: f64, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, A: [^]f64) ---
	chpr2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, Ap: [^]complex64) ---
	zhpr2 :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, Ap: [^]complex128) ---

	chbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint, beta: ^complex64, Y: [^]complex64, incY: blasint) ---
	zhbmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint, beta: ^complex128, Y: [^]complex128, incY: blasint) ---
	chpmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex64, Ap: [^]complex64, X: [^]complex64, incX: blasint, beta: ^complex64, Y: [^]complex64, incY: blasint) ---
	zhpmv :: proc(order: CBLAS_ORDER, Uplo: CBLAS_UPLO, N: blasint, alpha: ^complex128, Ap: [^]complex128, X: [^]complex128, incX: blasint, beta: ^complex128, Y: [^]complex128, incY: blasint) ---

	sgemm :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
	dgemm :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
	cgemm :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	cgemm3m :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	zgemm :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---
	zgemm3m :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---
	sgemmt :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
	dgemmt :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
	cgemmt :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	zgemmt :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

	ssymm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
	dsymm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
	csymm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	zsymm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

	ssyrk :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
	dsyrk :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
	csyrk :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	zsyrk :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---
	ssyr2k :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
	dsyr2k :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
	csyr2k :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	zsyr2k :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

	strmm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint) ---
	dtrmm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint) ---
	ctrmm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint) ---
	ztrmm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint) ---
	strsm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint) ---
	dtrsm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint) ---
	ctrsm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint) ---
	ztrsm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint) ---

	chemm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
	zhemm :: proc(Order: CBLAS_ORDER, Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

	cherk :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: f32, A: [^]complex64, lda: blasint, beta: f32, C: [^]complex64, ldc: blasint) ---
	zherk :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: f64, A: [^]complex128, lda: blasint, beta: f64, C: [^]complex128, ldc: blasint) ---
	cher2k :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: f32, C: [^]complex64, ldc: blasint) ---
	zher2k :: proc(Order: CBLAS_ORDER, Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: f64, C: [^]complex128, ldc: blasint) ---

	xerbla :: proc(p: blasint, rout: cstring, form: cstring, #c_vararg _: ..any) ---

	/*** BLAS extensions ***/
	saxpby :: proc(n: blasint, alpha: f32, x: [^]f32, incx: blasint, beta: f32, y: [^]f32, incy: blasint) ---
	daxpby :: proc(n: blasint, alpha: f64, x: [^]f64, incx: blasint, beta: f64, y: [^]f64, incy: blasint) ---
	caxpby :: proc(n: blasint, alpha: ^complex64, x: [^]complex64, incx: blasint, beta: ^complex64, y: [^]complex64, incy: blasint) ---
	zaxpby :: proc(n: blasint, alpha: ^complex128, x: [^]complex128, incx: blasint, beta: ^complex128, y: [^]complex128, incy: blasint) ---

	somatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: f32, a: [^]f32, clda: blasint, b: [^]f32, cldb: blasint) ---
	domatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: f64, a: [^]f64, clda: blasint, b: [^]f64, cldb: blasint) ---
	comatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: ^complex64, a: [^]complex64, clda: blasint, b: [^]complex64, cldb: blasint) ---
	zomatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: ^complex128, a: [^]complex128, clda: blasint, b: [^]complex128, cldb: blasint) ---

	simatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: f32, a: [^]f32, clda: blasint, cldb: blasint) ---
	dimatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: f64, a: [^]f64, clda: blasint, cldb: blasint) ---
	cimatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: ^complex64, a: [^]complex64, clda: blasint, cldb: blasint) ---
	zimatcopy :: proc(CORDER: CBLAS_ORDER, CTRANS: CBLAS_TRANSPOSE, crows: blasint, ccols: blasint, calpha: ^complex128, a: [^]complex128, clda: blasint, cldb: blasint) ---

	sgeadd :: proc(CORDER: CBLAS_ORDER, crows: blasint, ccols: blasint, calpha: f32, a: [^]f32, clda: blasint, cbeta: f32, _c: [^]f32, cldc: blasint) ---
	dgeadd :: proc(CORDER: CBLAS_ORDER, crows: blasint, ccols: blasint, calpha: f64, a: [^]f64, clda: blasint, cbeta: f64, _c: [^]f64, cldc: blasint) ---
	cgeadd :: proc(CORDER: CBLAS_ORDER, crows: blasint, ccols: blasint, calpha: ^complex64, a: [^]complex64, clda: blasint, cbeta: ^complex64, _c: [^]complex64, cldc: blasint) ---
	zgeadd :: proc(CORDER: CBLAS_ORDER, crows: blasint, ccols: blasint, calpha: ^complex128, a: [^]complex128, clda: blasint, cbeta: ^complex128, _c: [^]complex128, cldc: blasint) ---

	// EVERYTHING BELOW HERE IS NOT WRAPPED:
	sgemm_batch :: proc(Order: CBLAS_ORDER, TransA_array: [^]CBLAS_TRANSPOSE, TransB_array: [^]CBLAS_TRANSPOSE, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]f32, A_array: [^]^f32, lda_array: [^]blasint, B_array: [^]^f32, ldb_array: [^]blasint, beta_array: [^]f32, C_array: [^]^f32, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---
	dgemm_batch :: proc(Order: CBLAS_ORDER, TransA_array: [^]CBLAS_TRANSPOSE, TransB_array: [^]CBLAS_TRANSPOSE, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]f64, A_array: [^]^f64, lda_array: [^]blasint, B_array: [^]^f64, ldb_array: [^]blasint, beta_array: [^]f64, C_array: [^]^f64, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---
	cgemm_batch :: proc(Order: CBLAS_ORDER, TransA_array: [^]CBLAS_TRANSPOSE, TransB_array: [^]CBLAS_TRANSPOSE, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]complex64, A_array: [^][^]complex64, lda_array: [^]blasint, B_array: [^][^]complex64, ldb_array: [^]blasint, beta_array: [^]complex64, C_array: [^][^]complex64, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---
	zgemm_batch :: proc(Order: CBLAS_ORDER, TransA_array: [^]CBLAS_TRANSPOSE, TransB_array: [^]CBLAS_TRANSPOSE, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]complex128, A_array: [^][^]complex128, lda_array: [^]blasint, B_array: [^][^]complex128, ldb_array: [^]blasint, beta_array: [^]complex128, C_array: [^][^]complex128, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---

	/*** BFLOAT16 and INT8 extensions ***/
	/* convert float array to BFLOAT16 array by rounding */
	sbstobf16 :: proc(n: blasint, _in: [^]f32, incin: blasint, out: [^]bfloat16, incout: blasint) ---

	/* convert double array to BFLOAT16 array by rounding */
	sbdtobf16 :: proc(n: blasint, _in: [^]f64, incin: blasint, out: [^]bfloat16, incout: blasint) ---

	/* convert BFLOAT16 array to float array */
	sbf16tos :: proc(n: blasint, _in: [^]bfloat16, incin: blasint, out: [^]f32, incout: blasint) ---

	/* convert BFLOAT16 array to double array */
	dbf16tod :: proc(n: blasint, _in: [^]bfloat16, incin: blasint, out: [^]f64, incout: blasint) ---

	/* dot production of BFLOAT16 input arrays, and output as float */
	sbdot :: proc(n: blasint, x: [^]bfloat16, incx: blasint, y: [^]bfloat16, incy: blasint) -> f32 ---
	sbgemv :: proc(order: CBLAS_ORDER, trans: CBLAS_TRANSPOSE, m: blasint, n: blasint, alpha: f32, a: [^]bfloat16, lda: blasint, x: [^]bfloat16, incx: blasint, beta: f32, y: [^]f32, incy: blasint) ---
	sbgemm :: proc(Order: CBLAS_ORDER, TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, M: blasint, N: blasint, K: blasint, alpha: f32, A: [^]bfloat16, lda: blasint, B: [^]bfloat16, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
	sbgemm_batch :: proc(Order: CBLAS_ORDER, TransA_array: [^]CBLAS_TRANSPOSE, TransB_array: [^]CBLAS_TRANSPOSE, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]f32, A_array: [^]^bfloat16, lda_array: [^]blasint, B_array: [^]^bfloat16, ldb_array: [^]blasint, beta_array: [^]f32, C_array: [^]^f32, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---

}
