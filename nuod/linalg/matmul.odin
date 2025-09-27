package linalg

import "core:fmt"
import "base:intrinsics"
import md "../mdarray"
import "../logging"

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
