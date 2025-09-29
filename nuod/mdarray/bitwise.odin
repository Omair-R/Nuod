package mdarray

import "base:intrinsics"


@(private="file")
inner_bitwise_and :: #force_inline proc($T: typeid)-> proc(T, T, ..T)->T{
	return #force_inline proc (a: T, b: T, args: ..T) -> T { return a & b}
}
@(private="file")
inner_bitwise_or :: #force_inline proc($T: typeid)-> proc(T, T, ..T)->T{
	return #force_inline proc (a: T, b: T, args: ..T) -> T { return a | b}
}
@(private="file")
inner_bitwise_xor :: #force_inline proc($T: typeid)-> proc(T, T, ..T)->T{
	return #force_inline proc (a: T, b: T, args: ..T) -> T { return a ~ b}
}
@(private="file")
inner_bitwise_andnot :: #force_inline proc($T: typeid)-> proc(T, T, ..T)->T{
	return #force_inline proc (a: T, b: T, args: ..T) -> T { return a &~ b}
}
@(private="file")
inner_left_shift :: #force_inline proc($T: typeid)-> proc(T, T, ..T)->T{
	return #force_inline proc (a: T, b: T, args: ..T) -> T { return a &~ b}
}
@(private="file")
inner_right_shift :: #force_inline proc($T: typeid)-> proc(T, T, ..T)->T{
	return #force_inline proc (a: T, b: T, args: ..T) -> T { return a &~ b}
}
@(private="file")
inner_bitwise_comp :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_numeric(T) {
	return #force_inline proc(val: ^T)  { val^ = ~val^ }
}



bitwise_and_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_bitwise_and(T), allocator=allocator, location=location)
}


bitwise_and_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_bitwise_and(T), flip=false, allocator=allocator, location=location)
}


bitwise_and_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_bitwise_and(T), flip=true, allocator=allocator, location=location)
}


bitwise_or_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_bitwise_or(T), allocator=allocator, location=location)
}


bitwise_or_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_bitwise_or(T), flip=false, allocator=allocator, location=location)
}


bitwise_or_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_bitwise_or(T), flip=true, allocator=allocator, location=location)
}


bitwise_xor_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_bitwise_xor(T), allocator=allocator, location=location)
}


bitwise_xor_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_bitwise_xor(T), flip=false, allocator=allocator, location=location)
}


bitwise_xor_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_bitwise_xor(T), flip=true, allocator=allocator, location=location)
}


bitwise_andnot_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_bitwise_andnot(T), allocator=allocator, location=location)
}


bitwise_andnot_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_bitwise_andnot(T), flip=false, allocator=allocator, location=location)
}


bitwise_andnot_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_bitwise_andnot(T), flip=true, allocator=allocator, location=location)
}


left_shift_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_left_shift(T), allocator=allocator, location=location)
}


left_shift_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_left_shift(T), flip=false, allocator=allocator, location=location)
}


left_shift_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_left_shift(T), flip=true, allocator=allocator, location=location)
}


right_shift_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_right_shift(T), allocator=allocator, location=location)
}


right_shift_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_right_shift(T), flip=false, allocator=allocator, location=location)
}


right_shift_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_right_shift(T), flip=true, allocator=allocator, location=location)
}



outplace_bitwise_comp :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_integer(T) {
	return outplace_unary_map(mdarray, inner_sign(T), location)
}


inplace_bitwise_comp :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_integer(T) {
	return inplace_unary_map(mdarray, inner_sign(T), location)
}


bitwise_and :: proc{bitwise_and_arrays, bitwise_and_arrays_scalar, bitwise_and_scalar_array}
bitwise_or :: proc{bitwise_or_arrays, bitwise_or_arrays_scalar, bitwise_or_scalar_array}
bitwise_xor :: proc{bitwise_xor_arrays, bitwise_xor_arrays_scalar, bitwise_xor_scalar_array}
bitwise_andnot :: proc{bitwise_andnot_arrays, bitwise_andnot_arrays_scalar, bitwise_andnot_scalar_array}
left_shift :: proc{left_shift_arrays, left_shift_arrays_scalar, left_shift_scalar_array}
right_shift :: proc{right_shift_arrays, right_shift_arrays_scalar, right_shift_scalar_array}

o_bitwise_comp :: outplace_bitwise_comp
i_bitwise_comp :: inplace_bitwise_comp
