package linalg

import "core:slice"
import "core:math"
import "base:intrinsics"
import md "../mdarray"
import "../logging"
import lapacke "../lapacke"


validate_open_blas :: proc(
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (ok: bool) {
	
	md.validate_initialized(a, location) or_return


	if !lapacke.OPENBLAS_SUPPORTED {
		logging.error(
			.NotImplemented,
			"Decomposion operations are only implemented with openblas support.",
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

	return true
}


qr :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 q:md.MdArray(T, Nd),
	 r:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	validate_open_blas(a, allocator, location) or_return

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


full_svd :: proc(	
	$Nd: int,
	a: md.MdArray($T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 s:md.MdArray(T, Nd-1),
	 u:md.MdArray(T, Nd),
	 vt:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	validate_open_blas(a, allocator, location) or_return

	// Lapack will fill the array since it uses it as a work area.
	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	return _inner_svd(Nd, a_, .Full, allocator, location) 	
}


reduced_svd :: proc(	
	$Nd: int,
	a: md.MdArray($T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 s:md.MdArray(T, Nd-1),
	 u:md.MdArray(T, Nd),
	 vt:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	validate_open_blas(a, allocator, location) or_return

	// Lapack will fill the array since it uses it as a work area.
	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	return _inner_svd(Nd, a_, .Reduced, allocator, location) 	
}


svd_vals :: proc(	
	$Nd: int,
	a: md.MdArray($T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 s:md.MdArray(T, Nd-1),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 #optional_ok {

	validate_open_blas(a, allocator, location) or_return

	// Lapack will fill the array since it uses it as a work area.
	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	m:= a.shape[Nd-2]
	n:= a.shape[Nd-1]
	k:= min(m, n)

	m_b:= lapacke.blasint(m)
	n_b:= lapacke.blasint(n)

	s_shape : [Nd-1]int 
	s_shape[Nd-2] = k

	when Nd == 2 {
		s = md.make_mdarray(T, s_shape, allocator, location) or_return
		lapack_svd_wrapper(
			a.buffer, m_b, n_b,
			s.buffer, []T{},
			[]T{}, .Skip_UV
		) or_return
	} else { 
		for d in 0..<Nd-2{
			s_shape[d] = a.shape[d]
		}

		s = md.make_mdarray(T, s_shape, allocator, location) or_return

		a_sig:= m*n
		s_sig:= k

		a_s: []T
		s_s: []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
			s_s = s.buffer[i*s_sig: i*s_sig+s_sig]

			lapack_svd_wrapper(
				a_s, m_b, n_b,
				s_s, []T{},
				[]T{}, .Skip_UV
			) or_return
		}
	}
	return s, true 	
}


@private
_inner_svd :: proc(	
	$Nd: int,
	a: md.MdArray($T, Nd),
	mode: SVD_Mode,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 s:md.MdArray(T, Nd-1),
	 u:md.MdArray(T, Nd),
	 vt:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	m:= a.shape[Nd-2]
	n:= a.shape[Nd-1]
	k:= min(m, n)

	m_b:= lapacke.blasint(m)
	n_b:= lapacke.blasint(n)

	s_shape : [Nd-1]int 
	u_shape : [Nd]int
	v_shape : [Nd]int

	s_shape[Nd-2] = k
	
	#partial switch mode {
		case .Full:
			u_shape[Nd-2] = m
			u_shape[Nd-1] = m

			v_shape[Nd-2] = n
			v_shape[Nd-1] = n
		case .Reduced:
			u_shape[Nd-2] = m
			u_shape[Nd-1] = k

			v_shape[Nd-2] = k
			v_shape[Nd-1] = n
	}

	when Nd == 2 {
		s = md.make_mdarray(T, s_shape, allocator, location) or_return
		u = md.make_mdarray(T, u_shape, allocator, location) or_return
		vt = md.make_mdarray(T, v_shape, allocator, location) or_return
		lapack_svd_wrapper(
			a.buffer, m_b, n_b,
			s.buffer, u.buffer,
			vt.buffer, mode, location
		) or_return
	} else { 
		for d in 0..<Nd-2{
			s_shape[d] = a.shape[d]
			u_shape[d] = a.shape[d]
			vt_shape[d] = a.shape[d]
		}

		s = md.make_mdarray(T, s_shape, allocator, location) or_return
		u = md.make_mdarray(T, u_shape, allocator, location) or_return
		vt = md.make_mdarray(T, vt_shape, allocator, location) or_return

		a_sig:= m*n
		s_sig:= k
		u_sig:= m*m
		v_sig:= n*n

		a_s: []T
		s_s: []T
		u_s: []T
		v_s: []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
			s_s = s.buffer[i*s_sig: i*s_sig+s_sig]
			u_s = u.buffer[i*u_sig: i*u_sig+u_sig]
			v_s = vt.buffer[i*v_sig: i*v_sig+v_sig]

			lapack_svd_wrapper(
				a_s, m_b, n_b,
				s_s, u_s,
				v_s, mode
			) or_return
		}
	}

	return s, u, vt, true	
}


fill_eig_slices :: proc(
	e_vals: []$C,
	e_vecs: []C,
	wr, wi, vr : []$F,
) where intrinsics.type_is_float(F) || intrinsics.type_is_complex(C) {

	n:= len(e_vals)

	for i in 0..<n{
		e_vals[i] = complex(wr[i], wi[i])
	}

	for i in 0..<n{
		for j:=0; j<n;{
			if wi[j] == 0.0 {
				e_vecs[i*n+j] = complex(vr[i*n+j], 0)
				j += 1
			} else {
				e_vecs[i*n+j] = complex(vr[i*n+j], vr[i*n+j+1])
				e_vecs[i*n+j+1] = conj(e_vecs[i*n+j]) 
				j += 2
			}
		}
	}
}


eig_f32 :: proc(
	$Nd: int,
	a: md.MdArray(f32, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 eig_vals:md.MdArray(complex64, Nd-1),
	 eig_vecs:md.MdArray(complex64, Nd),
	 ok:bool,
) where Nd>=2 {
	return _inner_eig(Nd, complex64, a, allocator, location)
}


eig_f64 :: proc(
	$Nd: int,
	a: md.MdArray(f64, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 eig_vals:md.MdArray(complex128, Nd-1),
	 eig_vecs:md.MdArray(complex128, Nd),
	 ok:bool,
) where Nd>=2 {
	return _inner_eig(Nd, complex128, a, allocator, location)
}


eig :: proc{ eig_f32, eig_f64 }


_inner_eig :: proc(	
	$Nd: int,
	$C: typeid,
	a: md.MdArray($F, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 eig_vals:md.MdArray(C, Nd-1),
	 eig_vecs:md.MdArray(C, Nd),
	 ok:bool,
) where intrinsics.type_is_float(F) || intrinsics.type_is_complex(C), Nd>=2 {

	md.validate_initialized(a, location) or_return


	if !lapacke.OPENBLAS_SUPPORTED {
		logging.error(
			.NotImplemented,
			"Decomposion operations are only implemented with openblas support.",
			location,
		)
		return
	}

	when !(F == f32 || F == f64){
		logging.error(
			.ArguementError,
			"openblas routines do not support half precision routines.",
			location,
		)
		return
	}

	when (F == f32 && C != complex64) || (F == f64 && C != complex128){
		logging.error(
			.ArguementError,
			"the complex type must correspond to the float.",
			location,
		)
		return
	}
	
	n:= a.shape[Nd-1]
	if n != a.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"the eig values can only be computed for nonsymteric square matrices or stacks of matrices.",
			location=location,
		)
	}
	
	// Lapack will fill the array since it uses it as a work area.
	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	n_b:= lapacke.blasint(n)

	wr := make([]f64, n)
	wi := make([]f64, n)
	vr := make([]f64, n*n)

	defer {
		delete(wr)
		delete(wi)
		delete(vr)
	}

	e_vals_shape : [Nd-1]int
	e_vecs_shape : [Nd]int

	e_vals_shape[Nd-2] = n

	e_vecs_shape[Nd-1] = n
	e_vecs_shape[Nd-2] = n

	when Nd == 2 {
		eig_vals = md.make_mdarray(C, e_vals_shape, allocator, location) or_return
		eig_vecs = md.make_mdarray(C, e_vecs_shape, allocator, location) or_return
		lapack_eig_wrapper_f(
			a_.buffer, n_b,
			wr, wi, vr,
			true, location
		) or_return
		fill_eig_slices(eig_vals.buffer, eig_vecs.buffer, wr, wi, vr)
	} else { 
		for d in 0..<Nd-2{
			e_vals_shape[d] = a_.shape[d]
			e_vecs_shape[d] = a_.shape[d]
		}

		eig_vals = md.make_mdarray(T, e_vals_shape, allocator, location) or_return
		eig_vecs = md.make_mdarray(T, e_vecs_shape, allocator, location) or_return

		a_sig:= n*n
		e_vals_sig:= n
		e_vecs_sig:= n*n

		a_s: []T
		e_vals_s: []T
		e_vecs_s: []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a_.buffer[i*a_sig: i*a_sig+a_sig]
			e_vals_s = eig_vals.buffer[i*e_vals_sig: i*e_vals_sig+e_vals_sig]
			e_vecs_s = eig_vecs.buffer[i*e_vecs_sig: i*e_vecs_sig+e_vecs_sig]

			lapack_eig_wrapper_f(
				a_s, n_b,
				wr, wi, vr,
				true, location
			) or_return
			fill_eig_slices(e_vals_s, e_vecs_s, wr, wi, vr)
		}
	}

	return 
}


eigvals_f32 :: proc(
	$Nd: int,
	a: md.MdArray(f32, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 eig_vals:md.MdArray(complex64, Nd-1),
	 ok:bool,
) where Nd>=2 {
	return _inner_eigvals(Nd, complex64, a, allocator, location)
}


eigvals_f64 :: proc(
	$Nd: int,
	a: md.MdArray(f64, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 eig_vals:md.MdArray(complex128, Nd-1),
	 ok:bool,
) where Nd>=2 {
	return _inner_eigvals(Nd, complex128, a, allocator, location)
}


eigvals :: proc{ eigvals_f32, eigvals_f64 }


_inner_eigvals :: proc(	
	$Nd: int,
	$C: typeid,
	a: md.MdArray($F, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 eig_vals:md.MdArray(C, Nd-1),
	 ok:bool,
) where intrinsics.type_is_float(F) || intrinsics.type_is_complex(C), Nd>=2 {

	md.validate_initialized(a, location) or_return


	if !lapacke.OPENBLAS_SUPPORTED {
		logging.error(
			.NotImplemented,
			"Decomposion operations are only implemented with openblas support.",
			location,
		)
		return
	}

	when !(F == f32 || F == f64){
		logging.error(
			.ArguementError,
			"openblas routines do not support half precision routines.",
			location,
		)
		return
	}

	when (F == f32 && C != complex64) || (F == f64 && C != complex128){
		logging.error(
			.ArguementError,
			"the complex type must correspond to the float.",
			location,
		)
		return
	}
	
	n:= a.shape[Nd-1]
	if n != a.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"the eig values can only be computed for nonsymteric square matrices or stacks of matrices.",
			location=location,
		)
	}
	
	// Lapack will fill the array since it uses it as a work area.
	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	n_b:= lapacke.blasint(n)

	wr := make([]f64, n)
	wi := make([]f64, n)


	defer {
		delete(wr)
		delete(wi)
	}

	e_vals_shape : [Nd-1]int
	e_vals_shape[Nd-2] = n


	when Nd == 2 {
		eig_vals = md.make_mdarray(C, e_vals_shape, allocator, location) or_return
		lapack_eig_wrapper_f(
			a_.buffer, n_b,
			wr, wi, []F{},
			false, location
		) or_return
		for i in 0..<n{
			eig_vals.buffer[i] = complex(wr[i], wi[i])
		}
	} else { 
		for d in 0..<Nd-2{
			e_vals_shape[d] = a_.shape[d]
		}

		eig_vals = md.make_mdarray(T, e_vals_shape, allocator, location) or_return

		a_sig:= n*n
		e_vals_sig:= n

		a_s: []T
		e_vals_s: []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a_.buffer[i*a_sig: i*a_sig+a_sig]
			e_vals_s = eig_vals.buffer[i*e_vals_sig: i*e_vals_sig+e_vals_sig]

			lapack_eig_wrapper_f(
				a_s, n_b,
				wr, wi, []F{},
				false, location
			) or_return
			for i in 0..<n{
				e_vecs_s[i] = complex(wr[i], wi[i])
			}
		}
	}

	return 
}


