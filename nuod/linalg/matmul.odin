package linalg

import "core:fmt"
import "base:intrinsics"
import md "../mdarray"
import "../logging"
import "../cblas"


inner_product :: proc(	
	a: md.MdArray($T, 1),
	b: md.MdArray(T, 1),
	location := #caller_location,
) -> (
	 result:T,
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(b, location) or_return
	md.validate_shape_match(a, b, location) or_return

	when cblas.OPENBLAS_SUPPORTED && (T == f32 || T == f64 || T == complex64 || T == complex128) {
		if !a.is_view && !b.is_view {
			result = cblas_dot_wrapper(a.buffer, b.buffer) or_return
			return result, true
		}
	}

	for i in 0..<md.size(a){
		result += md.get_linear(a, i) * md.get_linear(b, i)	
	}

	return result, true
}


outer_product :: proc(	
	a: md.MdArray($T, 1),
	b: md.MdArray(T, 1),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, 2),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(b, location) or_return

	result = md.make_mdarray(T, [2]int{a.shape[0], b.shape[0]})

	for i in 0..<md.size(a){
		for j in 0..<md.size(b){
			val := md.get_ref(result, [2]int{i, j}, location) or_return
			val^ = md.get_linear(a, i) * md.get_linear(b, j)
		}
	}
	return result, true
}


kron_vector_product :: proc(	
	a: md.MdArray($T, 1),
	b: md.MdArray(T, 1),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, 1),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(b, location) or_return

	outer := outer_product(a, b, allocator, location) or_return
	defer md.free_mdarray(outer)
	result = md.flatten_copy(outer, allocator, location) or_return 
	return result, true
}


matmul :: proc(	
	a: md.MdArray($T, $Nd),
	b: md.MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T), Nd>=2 #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(b, location) or_return

	if a.shape[Nd-1] != b.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"Inner size of matrices is inconsistent.",
			location,
		)
		return 
	}

	a := a
	b := b

	a_is_view := a.is_view
	b_is_view := b.is_view
	if a_is_view {
		a = md.copy_array(a, allocator, location) or_return
	}
	defer if a_is_view do md.free_mdarray(a)
	if b_is_view {
		b = md.copy_array(b, allocator, location) or_return
	}
	defer if b_is_view do md.free_mdarray(b)

	when cblas.OPENBLAS_SUPPORTED &&
	(T == f32 || T == f64 || T == complex64 || T== complex128){
		return cblas_matmul(a, b, allocator, location)
	}

	a_v := md.expand_dim_view(Nd, a, axis=Nd, location=location) or_return
	b_v := md.expand_dim_view(Nd, b, axis=Nd-2, location=location) or_return

	
	f:: proc(a :T, b: T, args: ..T) -> T { return a * b }

	inter_mul := md.broadcast_map(a_v, b_v, f, allocator=allocator, location=location) 
	defer md.free_mdarray(inter_mul)


	result = md.dim_reduce_sum(Nd+1, inter_mul, Nd-1, allocator=allocator, location=location) or_return 

	return result, true
}


@(private="file")
cblas_matmul :: proc(	
	a: md.MdArray($T, $Nd),
	b: md.MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 #optional_ok {


	m:= a.shape[Nd-2]
	n:= b.shape[Nd-1]
	k:= a.shape[Nd-1]

	m_b:= cblas.blasint(m)
	n_b:= cblas.blasint(n)
	k_b:= cblas.blasint(k)

	result_shape : [Nd]int
	result_shape[Nd-2] = m
	result_shape[Nd-1] = n

	when Nd == 2 {
		result = md.make_mdarray(T, result_shape, allocator, location) or_return
		cblas_matmul_wrapper(
			a.buffer, b.buffer,
			m_b, n_b, k_b,
			result.buffer
		)  	or_return

	} else { 
		for d in 0..<Nd-2{
			if a.shape[d] != b.shape[d]{
				logging.error(
					.ArguementError,
					"Inconsistent shape for the stack of matrices provided",
					location,
				)
				return
			}
			result_shape[d] = a.shape[d]
		}

		result = md.make_mdarray(T, result_shape, allocator, location) or_return
		a_sig:= m*k
		b_sig:= n*k
		r_sig:= n*m

		a_s: []T
		b_s: []T
		c_out: []T

		for i in 0..<(md.size(result)/(r_sig)){
			c_out = result.buffer[i*r_sig: i*r_sig+r_sig]
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
			b_s = b.buffer[i*b_sig: i*b_sig+b_sig]

			cblas_matmul_wrapper(
				a_s, b_s,
				m_b, n_b, k_b,
				c_out
			)  	or_return
		}
	}

	return result, true
}


matvec :: proc(	
	a: md.MdArray($T, $Nd),
	v: md.MdArray(T, $Md),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, Md),
	 ok:bool,
) where intrinsics.type_is_numeric(T), (Nd-1)==Md #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(v, location) or_return

	if (v.shape[Md-1] != a.shape[Nd-1]){
		logging.error(
			.ArguementError,
			"Inconsistance shape with the length of provided arrays.",
			location = location,
		)
		return 
	}


	when Md != 1 do for d in 0..<Md-1 {
		if a.shape[d] != v.shape[d]{
			logging.error(
				.ArguementError,
				"Inconsistent shape for the stack of matrices provided",
				location,
			)
			return
		}
	}

	
	a := a
	v := v

	a_is_view := a.is_view
	v_is_view := v.is_view
	if a_is_view {
		a = md.copy_array(a, allocator, location) or_return
	}
	defer if a_is_view do md.free_mdarray(a)
	if v_is_view {
		v = md.copy_array(v, allocator, location) or_return
	}
	defer if v_is_view do md.free_mdarray(v)
	

	when cblas.OPENBLAS_SUPPORTED && (T == f32 || T == f64 || T == complex64 || T == complex128) {
		return cblas_matvec(a, v, allocator=allocator, location=location)
	}

	f:: proc(a :T, b: T, args: ..T) -> T { return a * b }

	v_r := md.expand_dim_view(Md, v, axis=Md-1, location=location) or_return

	inter_mul := md.broadcast_map(a , v_r, f, allocator=allocator, location=location) or_return
	defer md.free_mdarray(inter_mul)

	result = md.dim_reduce_sum(Nd, inter_mul, Md, allocator=allocator, location=location) or_return 
	return result, true
}


vecmat :: proc(	
	v: md.MdArray($T, $Md),
	a: md.MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, Md),
	 ok:bool,
) where intrinsics.type_is_numeric(T), (Nd-1)==Md #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(v, location) or_return

	if (v.shape[Md-1] != a.shape[Nd-2]){
		logging.error(
			.ArguementError,
			"Inconsistance shape with the length of provided arrays.",
			location = location,
		)
		return 
	}


	when Md != 1 do for d in 0..<Md-1 {
		if a.shape[d] != v.shape[d]{
			logging.error(
				.ArguementError,
				"Inconsistent shape for the stack of matrices provided",
				location,
			)
			return
		}
	}

	
	a := a
	v := v

	a_is_view := a.is_view
	v_is_view := v.is_view
	if a_is_view {
		a = md.copy_array(a, allocator, location) or_return
	}
	defer if a_is_view do md.free_mdarray(a)
	if v_is_view {
		v = md.copy_array(v, allocator, location) or_return
	}
	defer if v_is_view do md.free_mdarray(v)
	

	when cblas.OPENBLAS_SUPPORTED && (T == f32 || T == f64 || T == complex64 || T == complex128) {
		return cblas_matvec(a, v, transpose_a=true, allocator=allocator, location=location)
	}

	f:: proc(a :T, b: T, args: ..T) -> T { return a * b }

	v_r := md.expand_dim_view(Md, v, axis=Md, location=location) or_return

	inter_mul := md.broadcast_map(a , v_r, f, allocator=allocator, location=location) or_return
	defer md.free_mdarray(inter_mul)

	result = md.dim_reduce_sum(Nd, inter_mul, Md-1, allocator=allocator, location=location) or_return 
	return result, true
}


@(private="file")
cblas_matvec :: proc(	
	a: md.MdArray($T, $Nd),
	v: md.MdArray(T, $Md),
	transpose_a:=false,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, Md),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), (Nd-1)==Md #optional_ok {
	m := a.shape[Nd-2]
	n := a.shape[Nd-1]

	result_shape := v.shape

	result_shape[Md-1] = transpose_a? n : m

	result = md.make_mdarray(T, result_shape, allocator, location) or_return
	when Nd == 2 {
		cblas_matvec_wrapper(
			a.buffer,
			v.buffer,
			cblas.blasint(m),
			cblas.blasint(n),
			result.buffer,
			transpose_a = transpose_a,
		) or_return
		return result, true
	}

	a_sig:= m*n
	v_sig:= n
	r_sig:= m

	a_s: []T
	v_s: []T
	w_out: []T

	m_b:= cblas.blasint(m)
	n_b:= cblas.blasint(n)

	for i in 0..<(md.size(result)/r_sig){
			w_out = result.buffer[i*r_sig: i*r_sig+r_sig]
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
			v_s = v.buffer[i*v_sig: i*v_sig+v_sig]

			cblas_matvec_wrapper(
				a_s,
				v_s,
				m_b, n_b,
				w_out,
				transpose_a = transpose_a,
			) or_return
	}

	return result, true
}
