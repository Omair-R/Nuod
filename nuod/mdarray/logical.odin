package mdarray

import "base:intrinsics"
import "core:math"


@(private="file")
inner_logical_and :: #force_inline proc (a: bool, b: bool) -> bool { return a && b}
@(private="file")
inner_logical_or :: #force_inline proc (a: bool, b: bool) -> bool { return a || b}



logical_and_arrays :: proc(	
	a: MdArray(bool, $Nd),
	b: MdArray(bool, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return element_wise_map(a, b, inner_logical_and, allocator, location)
}


logical_and_arrays_scalar :: proc(	
	a: MdArray(bool, $Nd),
	b: bool,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return scalar_map(a, b, inner_logical_and, false, allocator, location)
}


logical_and_scalar_array :: proc(	
	a: bool,
	b: MdArray(bool, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return scalar_map(b, a, inner_logical_and, true, allocator, location)
}


logical_or_arrays :: proc(	
	a: MdArray(bool, $Nd),
	b: MdArray(bool, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return element_wise_map(a, b, inner_logical_or, allocator, location)
}


logical_or_arrays_scalar :: proc(	
	a: MdArray(bool, $Nd),
	b: bool,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return scalar_map(a, b, inner_logical_or, false, allocator, location)
}


logical_or_scalar_array :: proc(	
	a: bool,
	b: MdArray(bool, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return scalar_map(b, a, inner_logical_or, true, allocator, location)
}


all :: proc(
	mdarray: MdArray(bool, $Nd),
	location := #caller_location,
) -> (
	accum:bool, ok:bool
) #optional_ok {

	validate_initialized(mdarray, location) or_return

	for i in 0..<size(mdarray){
		if !get_linear(mdarray, i) {
			return false, true
		}
	}
	return true, true
}

any :: proc(
	mdarray: MdArray(bool, $Nd),
	location := #caller_location,
) -> (
	accum:bool, ok:bool
) #optional_ok {

	validate_initialized(mdarray, location) or_return

	for i in 0..<size(mdarray){
		if get_linear(mdarray, i) {
			return true, true
		}
	}
	return false, true
}
