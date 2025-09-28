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

	if a.shape != b.shape{
		logging.error(.ArguementError, "the length of the vectors should be equal.", location)
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

	a_shape : [Nd+1]int
	b_shape : [Nd+1]int

	offset := 0
	for i in 0..<Nd{
		if i == Nd-2{
			b_shape[i] = 1
			offset = 1
		}
		i_b:= i+offset
		a_shape[i] = a.shape[i]
		b_shape[i_b] = b.shape[i]
	}
	a_shape[Nd] = 1


	a_v := md.reshape_view(a, a_shape, location) or_return
	b_v := md.reshape_view(b, b_shape, location) or_return

	a_shape[Nd] = b_shape[Nd]
	b_shape[Nd-2] = a_shape[Nd-2]


	a_b := md.broadcast_to(a_v, a_shape, allocator, location) or_return
	b_b := md.broadcast_to(b_v, b_shape, allocator, location) or_return
	defer md.free_mdarray(a_b)
	defer md.free_mdarray(b_b)


	inter_mul := md.mul(a_b, b_b, allocator, location) or_return
	defer md.free_mdarray(inter_mul)


	result = md.dim_reduce_sum(Nd+1, inter_mul, Nd-1, allocator=allocator, location=location) or_return 

	return result, true
}


// The name is temp.
cblas_matmul :: proc(	
	a: md.MdArray($T, $Nd),
	b: md.MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T), Nd>=2 #optional_ok {

	md.validate_initialized(a, location) or_return
	md.validate_initialized(b, location) or_return

	if a.is_view || b.is_view{
		logging.error( //TODO
			.NotImplemented,
			"BLAS-based matrix multiplication isn't supported for views, yet.",
			location,
		)
		return 
	}

	m:= a.shape[Nd-2]
	n:= a.shape[Nd-1]
	k:= a.shape[Nd-1]

	if k != b.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"Inner size of matrices is inconsistent.",
			location,
		)
		return 
	}

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

	} else { // TODO: test this part
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

		m_b:= cblas.blasint(m)
		n_b:= cblas.blasint(n)
		k_b:= cblas.blasint(k)

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
