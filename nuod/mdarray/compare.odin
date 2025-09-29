package mdarray

import "base:intrinsics"
import "core:math"


@(private="file")
inner_equal :: #force_inline proc($T: typeid)-> proc(T, T, ..bool)->bool{
	return #force_inline proc (a: T, b: T, args: ..bool) -> bool { return a == b}
}
@(private="file")
inner_not_equal :: #force_inline proc($T: typeid)-> proc(T, T, ..bool)->bool{
	return #force_inline proc (a: T, b: T, args: ..bool) -> bool { return a != b}
}
@(private="file")
inner_greater :: #force_inline proc($T: typeid)-> proc(T, T, ..bool)->bool{
	return #force_inline proc (a: T, b: T, args: ..bool) -> bool { return a > b}
}
@(private="file")
inner_greater_equal :: #force_inline proc($T: typeid)-> proc(T, T, ..bool)->bool{
	return #force_inline proc (a: T, b: T, args: ..bool) -> bool { return a >= b}
}
@(private="file")
inner_less :: #force_inline proc($T: typeid)-> proc(T, T, ..bool)->bool{
	return #force_inline proc (a: T, b: T, args: ..bool) -> bool { return a < b}
}
@(private="file")
inner_less_equal :: #force_inline proc($T: typeid)-> proc(T, T, ..bool)->bool{
	return #force_inline proc (a: T, b: T, args: ..bool) -> bool { return a <= b}
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
	return element_wise_map(a, b, inner_equal(T), allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_equal(T), flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_bitwise_and(T), flip=true, allocator=allocator, location=location)
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
	return element_wise_map(a, b, inner_not_equal(T), allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_not_equal(T), flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_not_equal(T), flip=true, allocator=allocator, location=location)
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
	return element_wise_map(a, b, inner_greater(T), allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_greater(T), flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_greater(T), flip=true, allocator=allocator, location=location)
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
	return element_wise_map(a, b, inner_greater_equal(T), allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_greater_equal(T), flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_greater_equal(T), flip=true, allocator=allocator, location=location)
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
	return element_wise_map(a, b, inner_less(T), allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_less(T), flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_less(T), flip=true, allocator=allocator, location=location)
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
	return element_wise_map(a, b, inner_less_equal(T), allocator=allocator, location=location)
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
	return scalar_map(a, b, inner_less_equal(T), flip=false, allocator=allocator, location=location)
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
	return scalar_map(b, a, inner_less_equal(T), flip=true, allocator=allocator, location=location)
}


equal :: proc{equal_arrays, equal_arrays_scalar, equal_scalar_array}
not_equal :: proc{not_equal_arrays, not_equal_arrays_scalar, not_equal_scalar_array}
greater :: proc{greater_arrays, greater_arrays_scalar, greater_scalar_array}
greater_equal :: proc{greater_equal_arrays, greater_equal_arrays_scalar, greater_equal_scalar_array}
less :: proc{less_arrays, less_arrays_scalar, less_scalar_array}
less_equal :: proc{less_equal_arrays, less_equal_arrays_scalar, less_equal_scalar_array}
