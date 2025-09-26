package mdarray

import "base:intrinsics"
import "core:math"


element_wise_map :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	f: proc(T, T)->$S,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(S, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) {
	
	validate_initialized(a, location) or_return
	validate_initialized(b, location) or_return

	result = make_mdarray(T, a.shape, allocator, location) or_return
	
	for i in 0..<size(a){
		result.buffer[i] = f(get_linear(a, i), get_linear(b, i))		
	}
	return result, true
}


scalar_map :: proc(
	a: MdArray($T, $Nd),
	b: T,
	f: proc(T, T)->$S,
	flip:= false,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(S, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) {
	
	validate_initialized(a, location) or_return

	result = make_mdarray(S, a.shape, allocator, location) or_return

	if flip {
		for i in 0..<size(a){
			result.buffer[i] = f(b, get_linear(a, i))		
		}
		return result, true
	}

	for i in 0..<size(a){
		result.buffer[i] = f(get_linear(a, i), b)		
	}
	return result, true
}



@(private="file")
inner_add :: #force_inline  proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return a + b}
}
@(private="file")
inner_subtract :: #force_inline  proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return a - b}
}
@(private="file")
inner_mul :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return a * b}
}
@(private="file")
inner_div :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return a / b}
}
@(private="file")
inner_mod :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return a % b}
}
@(private="file")
inner_remainder :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return a %% b}
}
@(private="file")
inner_max :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.max(a, b)}
}
@(private="file")
inner_min :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.min(a, b)}
}
@(private="file")
inner_logaddexp :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.ln(math.exp(a)+math.exp(b))}
}
@(private="file")
inner_gcd :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.gcd(a, b)}
}
@(private="file")
inner_lcm :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.lcm(a, b)}
}
@(private="file")
inner_hypot :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.sqrt(a*a, b*b)}
}
@(private="file")
inner_atan2 :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T { return math.atan2(a, b)}
}
@(private="file")
inner_heaviside :: #force_inline proc($T: typeid)-> proc(T, T)->T{
	return #force_inline proc (a: T, b: T) -> T {
		return a<0? 0 : a>0? 1 : b
	}
}



add_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_add(T), allocator, location)
}


subtract_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_subtract(T), allocator, location)
}


mul_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_mul(T), allocator, location)
}


div_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_div(T), allocator, location)
}


mod_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_int(T) #optional_ok {
	return element_wise_map(a, b, inner_mod(T), allocator, location)
}


min_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_min(T), allocator, location)
}


max_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_max(T), allocator, location)
}


logaddexp_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return element_wise_map(a, b, inner_logaddexp(T), allocator, location)
}


gcd_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_gcd(T), allocator, location)
}


lcm_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_lcm(T), allocator, location)
}


hypot_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_hypot(T), allocator, location)
}


atan2_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_atan2(T), allocator, location)
}


heaviside_arrays :: proc(	
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_heaviside(T), allocator, location)
}


add_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_add(T), false, allocator, location)
}


subtract_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_subtract(T), false, allocator, location)
}


mul_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_mul(T), false, allocator, location)
}


div_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_div(T), false, allocator, location)
}


mod_arrays_scalar :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_int(T) #optional_ok {
	return scalar_map(a, b, inner_mod(T), false, allocator, location)
}


min_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_min(T), false, allocator, location)
}


max_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_max(T), false, allocator, location)
}


logaddexp_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(a, b, inner_logaddexp(T), false, allocator, location)
}


gcd_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_gcd(T), false, allocator, location)
}


lcm_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_lcm(T), false, allocator, location)
}


hypot_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_hypot(T), false, allocator, location)
}


atan2_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_atan2(T), false, allocator, location)
}


heaviside_arrays_scalar  :: proc(	
	a: MdArray($T, $Nd),
	b: T,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_heaviside(T), false, allocator, location)
}


add_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_add(T), true, allocator, location)
}


subtract_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_subtract(T), true, allocator, location)
}


mul_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_mul(T), true, allocator, location)
}


div_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_div(T), true, allocator, location)
}


mod_scalar_array :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_int(T) #optional_ok {
	return scalar_map(b, a, inner_mod(T), true, allocator, location)
}


min_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_min(T), true, allocator, location)
}


max_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_max(T), true, allocator, location)
}


logaddexp_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(b, a, inner_logaddexp(T), true, allocator, location)
}


gcd_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(b, a, inner_gcd(T), true, allocator, location)
}


lcm_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(b, a, inner_lcm(T), true, allocator, location)
}


hypot_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(b, a, inner_hypot(T), true, allocator, location)
}


atan2_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(b, a, inner_atan2(T), true, allocator, location)
}


heaviside_scalar_array  :: proc(	
	a: $T,
	b: MdArray(T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_heaviside(T), true, allocator, location)
}


add :: proc{add_arrays, add_arrays_scalar, add_scalar_array}
subtract :: proc{subtract_arrays, subtract_arrays_scalar, subtract_scalar_array}
mul :: proc{mul_arrays, mul_arrays_scalar, mul_scalar_array}
div :: proc{div_arrays, div_arrays_scalar, div_scalar_array}
mod :: proc{mod_arrays, mod_arrays_scalar, mod_scalar_array}
minimum :: proc{min_arrays, min_arrays_scalar, min_scalar_array}
maximum :: proc{max_arrays, max_arrays_scalar, max_scalar_array}
logaddexp :: proc{logaddexp_arrays, logaddexp_arrays_scalar, logaddexp_scalar_array}
gcd :: proc{gcd_arrays, gcd_arrays_scalar, gcd_scalar_array}
lcm :: proc{lcm_arrays, lcm_arrays_scalar, lcm_scalar_array}
hypot :: proc{hypot_arrays, hypot_arrays_scalar, hypot_scalar_array}
atan2 :: proc{atan2_arrays, atan2_arrays_scalar, atan2_scalar_array}
heaviside :: proc{heaviside_arrays, heaviside_arrays_scalar, heaviside_scalar_array}


