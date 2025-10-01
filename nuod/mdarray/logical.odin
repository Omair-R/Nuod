package mdarray

import "base:intrinsics"
import "core:math"

import "../logging"

@(private="file")
inner_logical_and :: #force_inline proc (a: bool, b: bool, args:..bool) -> bool { return a && b}
@(private="file")
inner_logical_or :: #force_inline proc (a: bool, b: bool, args:..bool) -> bool { return a || b}



logical_and_arrays :: proc(	
	a: MdArray(bool, $Nd),
	b: MdArray(bool, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) #optional_ok {
	return element_wise_map(a, b, inner_logical_and, allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_logical_and, flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_logical_and, flip=true, allocator=allocator, location=location)
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
	return element_wise_map(a, b, inner_logical_or, allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_logical_or, flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_logical_or, flip=true, allocator=allocator, location=location)
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


@(private = "file")
inner_is_close :: #force_inline proc($T: typeid) -> (
	proc(_: T, _: T, _: ..T) -> bool,
) where intrinsics.type_is_float(T) {
	return #force_inline proc(a: T, b: T, args: ..T) -> bool {
		return abs(a - b) <= (args[1] + args[0] * abs(b)) // args[0] is rtol - args[1] is atol
	}
}


get_default_tol :: #force_inline proc "contextless"($T: typeid) -> (
	rtol, atol: T,
	ok: bool,
){
	when T == f32 || T == f64 || T == complex64 || T==complex128 {
		rtol = T(1e-7)
		atol = T(1e-8)
	} else when T == f16 || T == complex32 {
		rtol = T(1e-3)
		atol = T(1e-4)
	} else {
		logging.error(
			.ArguementError,
			"Unsupported type.",
			location
		)
		return rtol, atol, false
	}

	return rtol, atol, true
}


@private
is_close_default :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T)  #optional_ok {

	rtol, atol := get_default_tol(T) or_return

	return is_close_with_args(a, b, rtol, atol, allocator=allocator, location=location)
}


@private
is_close_with_args :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	rtol: T,
	atol: T, 
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T)  #optional_ok {
	validate_initialized(a, location) or_return
	validate_initialized(b, location) or_return

	validate_shape_match(a, b, location=location) or_return

	return element_wise_map(a, b, inner_is_close(T), rtol, atol, allocator=allocator, location=location)
}

is_close :: proc{is_close_default, is_close_with_args}


all_close :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	location := #caller_location,
) -> (
	 result: bool,
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T)  #optional_ok {

	rtol, atol := get_default_tol(T) or_return

	return all_close_with_args(a, b, rtol, atol, location=location)
}


all_close_with_args :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	rtol: T,
	atol: T, 
	location := #caller_location,
) -> (
	 result: bool,
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T)  #optional_ok {

	validate_initialized(a, location) or_return
	validate_initialized(b, location) or_return

	validate_shape_match(a, b, location=location) or_return

	is_close_unwrapped := inner_is_close(T)

	for i in 0..<size(a){
		a_val := get_linear(a, i)
		b_val := get_linear(b, i)

		if !is_close_unwrapped(a_val, b_val, rtol, atol) do return false, true
	}

	return true, true
}
