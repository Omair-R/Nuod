package mdarray

import "base:intrinsics"
import "core:math"


@(private="file")
inner_equal :: #force_inline proc($T: typeid)-> proc(T, T)->bool{
	return #force_inline proc (a: T, b: T) -> bool { return a == b}
}
@(private="file")
inner_not_equal :: #force_inline proc($T: typeid)-> proc(T, T)->bool{
	return #force_inline proc (a: T, b: T) -> bool { return a != b}
}
@(private="file")
inner_greater :: #force_inline proc($T: typeid)-> proc(T, T)->bool{
	return #force_inline proc (a: T, b: T) -> bool { return a > b}
}
@(private="file")
inner_greater_equal :: #force_inline proc($T: typeid)-> proc(T, T)->bool{
	return #force_inline proc (a: T, b: T) -> bool { return a >= b}
}
@(private="file")
inner_less :: #force_inline proc($T: typeid)-> proc(T, T)->bool{
	return #force_inline proc (a: T, b: T) -> bool { return a < b}
}
@(private="file")
inner_less_equal :: #force_inline proc($T: typeid)-> proc(T, T)->bool{
	return #force_inline proc (a: T, b: T) -> bool { return a <= b}
}


equal_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_equal(T), allocator, location)
}


equal_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_equal(T), false, allocator, location)
}


equal_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_bitwise_and(T), true, allocator, location)
}


not_equal_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_not_equal(T), allocator, location)
}


not_equal_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_not_equal(T), false, allocator, location)
}


not_equal_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_not_equal(T), true, allocator, location)
}


greater_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_greater(T), allocator, location)
}


greater_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_greater(T), false, allocator, location)
}


greater_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_greater(T), true, allocator, location)
}


greater_equal_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_greater_equal(T), allocator, location)
}


greater_equal_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_greater_equal(T), false, allocator, location)
}


greater_equal_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_greater_equal(T), true, allocator, location)
}


less_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_less(T), allocator, location)
}


less_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_less(T), false, allocator, location)
}


less_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_less(T), true, allocator, location)
}


less_equal_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_less_equal(T), allocator, location)
}


less_equal_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_less_equal(T), false, allocator, location)
}


less_equal_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(bool, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_less_equal(T), true, allocator, location)
}


equal :: proc{equal_arrays, equal_arrays_scalar, equal_scalar_array}
not_equal :: proc{not_equal_arrays, not_equal_arrays_scalar, not_equal_scalar_array}
greater :: proc{greater_arrays, greater_arrays_scalar, greater_scalar_array}
greater_equal :: proc{greater_equal_arrays, greater_equal_arrays_scalar, greater_equal_scalar_array}
less :: proc{less_arrays, less_arrays_scalar, less_scalar_array}
less_equal :: proc{less_equal_arrays, less_equal_arrays_scalar, less_equal_scalar_array}
