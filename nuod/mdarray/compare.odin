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

/*
Perform an element-wise equal operation on two different arrays.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise equal operation on an arrays and a scalar.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise equal operation on a scalar and an array.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise not equal operation on two different arrays.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise not equal operation on an arrays and a scalar.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise not equal operation on a scalar and an array.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise greater than operation on two different arrays.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise greater than operation on an arrays and a scalar.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise greater than operation on a scalar and an array.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise greater than or equal operation on two different arrays.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise greater than or equal operation on an arrays and a scalar.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise greater than or equal operation on a scalar and an array.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise less than operation on two different arrays.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise less than operation on an arrays and a scalar.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise less than operation on a scalar and an array.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise less than or equal operation on two different arrays.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise less than or equal operation on an arrays and a scalar.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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

/*
Perform an element-wise less than or equal operation on a scalar and an array.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
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
