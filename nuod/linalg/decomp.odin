package linalg

import "base:intrinsics"
import md "../mdarray"
import "../logging"
import lapacke "../lapacke"


qr :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 q:md.MdArray(T, Nd),
	 r:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	md.validate_initialized(a, location) or_return


	if !lapacke.OPENBLAS_SUPPORTED {
		logging.error(
			.NotImplemented,
			"QR decomposion is only implemented with openblas support.",
			location,
		)
		return
	}

	when !(T == f32 || T == f64 || T == complex64 || T== complex128){
		logging.error(
			.ArguementError,
			"openblas routines do not support half precision routines.",
			location,
		)
		return
	}

	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	return lapacke_qr(a_, allocator, location) 	
}


@(private="file")
lapacke_qr :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 q:md.MdArray(T, Nd),
	 r:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {


	m:= a.shape[Nd-2]
	n:= a.shape[Nd-1]
	k:= min(m, n)

	m_b:= lapacke.blasint(m)
	n_b:= lapacke.blasint(n)

	r_shape : [Nd]int
	r_shape[Nd-2] = k
	r_shape[Nd-1] = n

	q_shape : [Nd]int
	q_shape[Nd-2] = m
	q_shape[Nd-1] = k


	when Nd == 2 {
		r = md.make_mdarray(T, r_shape, allocator, location) or_return
		q = md.make_mdarray(T, q_shape, allocator, location) or_return
		lapack_qr_wrapper(
			a.buffer,
			m_b, n_b,
			r.buffer, q.buffer,
			allocator,
			location
		) or_return
	} else { 
		for d in 0..<Nd-2{
			r_shape[d] = a.shape[d]
			q_shape[d] = a.shape[d]
		}

		r = md.make_mdarray(T, r_shape, allocator, location) or_return
		q = md.make_mdarray(T, q_shape, allocator, location) or_return

		a_sig:= m*n
		r_sig:= k*n
		q_sig:= m*k

		a_s: []T
		r_s: []T
		q_s: []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
			r_s = r.buffer[i*r_sig: i*r_sig+r_sig]
			q_s = q.buffer[i*q_sig: i*q_sig+q_sig]

			lapack_qr_wrapper(
				a_s,
				m_b, n_b,
				r_s, q_s,
				allocator,
				location
			) or_return
		}
	}

	return q, r, true
}

