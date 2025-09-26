package mdarray

import "base:intrinsics"
import "core:math"

import "../logging"


outplace_unary_map :: proc(
	mdarray: MdArray($T, $Nd),
	f: proc(^T),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T){
	
	result = copy_array(mdarray, allocator=allocator, location=location) or_return

	for &val in result.buffer {
		f(&val)
	}
	for i in 0..<size(mdarray){		
		f(get_linear_ref(mdarray, i))
	}

	return result, true
}

	
inplace_unary_map :: proc(
	mdarray: MdArray($T, $Nd),
	f: proc(^T), 	
	location := #caller_location,
) -> (
	 ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T){
	
	validate_initialized(mdarray, location) or_return
	
	for i in 0..<size(mdarray){		
		f(get_linear_ref(mdarray, i))
	}
	
	return true
}

// Sign operations
@(private="file")
inner_sign :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_numeric(T){
	return  #force_inline proc(val: ^T) { val^ = T(int(val^>0) - int(val^<0)) }
}
@(private="file")
inner_neg :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_numeric(T) {
	return #force_inline proc(val: ^T)  { val^ = -val^ }
}
@(private="file")
inner_abs :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_numeric(T) {
	return #force_inline proc(val: ^T)  { val^ = math.abs(val^) }
}

// Power operations
@(private="file")
inner_sq :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_numeric(T) {
	return #force_inline proc(val: ^T)  { val^ = val^ * val^ }
}
@(private="file")
inner_sqrt :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.sqrt(val^) }
}
@(private="file")
inner_cbrt :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.pow(val^, cast(T)(1.0/3.0)) }
}
@(private="file")
inner_exp :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.exp(val^) }
}
@(private="file")
inner_exp2 :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.pow(cast(T)2, val^) }
}
@(private="file")
inner_ln :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.ln(val^) }
}
@(private="file")
inner_log2 :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.log2(val^) }
}
@(private="file")
inner_log10 :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.log10(val^) }
}
@(private="file")
inner_log1p :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.log1p(val^) }
}
@(private="file")
inner_reciprocal :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T) {
	return #force_inline proc(val: ^T)  { val^ = 1/val^ }
}

//Trigonometric functions
@(private="file")
inner_sin :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.sin(val^) }
}
@(private="file")
inner_cos :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.cos(val^) }
}
@(private="file")
inner_tan :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.tan(val^) }
}
@(private="file")
inner_asin :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.asin(val^) }
}
@(private="file")
inner_acos :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.acos(val^) }
}
@(private="file")
inner_atan :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.atan(val^) }
}
@(private="file")
inner_degrees :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.to_degrees(val^) }
}
@(private="file")
inner_radians :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.to_radians(val^) }
}

//Hyperbolic

@(private="file")
inner_sinh :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.sinh(val^) }
}
@(private="file")
inner_cosh :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.cosh(val^) }
}
@(private="file")
inner_tanh :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.tanh(val^) }
}
@(private="file")
inner_asinh :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.asinh(val^) }
}
@(private="file")
inner_acosh :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.acosh(val^) }
}
@(private="file")
inner_atanh :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) { val^ = math.atanh(val^) }
}

// Complex
@(private="file")
inner_conj :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_complex(T) {
	return #force_inline proc(val: ^T)  { val^ = conj(val^) }
}

// Misc
@(private="file")
inner_sinc :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T){
	return #force_inline proc(val: ^T) {
		pi_val := math.PI * val^
		val^ = math.sin(pi_val)/pi_val
	}
}

outplace_sign :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return outplace_unary_map(mdarray, inner_sign(T), allocator, location)
}


inplace_sign :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_sign(T), location)
}


outplace_neg :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return outplace_unary_map(mdarray, inner_neg(T), allocator, location)
}


inplace_neg :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_neg(T), location)
}


outplace_abs :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return outplace_unary_map(mdarray, inner_abs(T), allocator, location)
}


inplace_abs :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_abs(T), location)
}

// ----

outplace_sq :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return outplace_unary_map(mdarray, inner_sq(T), allocator, location)
}


inplace_sq :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_sq(T), location)
}


outplace_sqrt :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_sqrt(T), allocator, location)
}


inplace_sqrt :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sqrt(T), location)
}


outplace_cbrt :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_cbrt(T), allocator, location)
}


inplace_cbrt :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_cbrt(T), location)
}


outplace_exp :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_exp(T), allocator, location)
}


inplace_exp :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_exp(T), location)
}


outplace_exp2 :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_exp2(T), allocator, location)
}


inplace_exp2 :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_exp2(T), location)
}


outplace_ln :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_ln(T), allocator, location)
}


inplace_ln :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_ln(T), location)
}


outplace_log2 :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_log2(T), allocator, location)
}


inplace_log2 :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_log2(T), location)
}


outplace_log10 :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_log10(T), allocator, location)
}


inplace_log10 :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_log10(T), location)
}


outplace_log1p :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_log1p(T), allocator, location)
}


inplace_log1p :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_log1p(T), location)
}


outplace_reciprocal :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return outplace_unary_map(mdarray, inner_reciprocal(T), allocator, location)
}


inplace_reciprocal :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_reciprocal(T), location)
}


// -----

outplace_sin :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_sin(T), allocator, location)
}


inplace_sin :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sin(T), location)
}


outplace_cos :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_cos(T), allocator, location)
}


inplace_cos :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_cos(T), location)
}


outplace_tan :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_tan(T), allocator, location)
}


inplace_tan :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_tan(T), location)
}


outplace_asin :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_asin(T), allocator, location)
}


inplace_asin :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_asin(T), location)
}


outplace_acos :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_acos(T), allocator, location)
}


inplace_acos :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_acos(T), location)
}


outplace_atan :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_atan(T), allocator, location)
}


inplace_atan :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_atan(T), location)
}


outplace_degrees :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_degrees(T), allocator, location)
}


inplace_degrees :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_degrees(T), location)
}


outplace_radians :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_radians(T), allocator, location)
}


inplace_radians :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_radians(T), location)
}


// ------

outplace_sinh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_sinh(T), allocator, location)
}


inplace_sinh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sinh(T), location)
}


outplace_cosh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_cosh(T), allocator, location)
}


inplace_cosh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_cosh(T), location)
}


outplace_tanh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_tanh(T), allocator, location)
}


inplace_tanh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_tanh(T), location)
}


outplace_asinh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_asinh(T), allocator, location)
}


inplace_asinh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_asinh(T), location)
}


outplace_acosh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_acosh(T), allocator, location)
}


inplace_acosh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_acosh(T), location)
}


outplace_atanh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_atanh(T), allocator, location)
}


inplace_atanh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_atanh(T), location)
}


// ------

outplace_conj :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_complex(T) {
	return outplace_unary_map(mdarray, inner_conj(T), allocator, location)
}


inplace_conj :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_complex(T) {
	return inplace_unary_map(mdarray, inner_conj(T), location)
}


outplace_sinc :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) {
	return outplace_unary_map(mdarray, inner_sinc(T), allocator, location)
}


inplace_sinc :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sinc(T), location)
}


// Aliases

i_sign :: inplace_sign
i_neg :: inplace_neg
i_abs :: inplace_abs

i_sq :: inplace_sq
i_sqrt :: inplace_sqrt
i_cbrt :: inplace_cbrt
i_exp :: inplace_exp
i_exp2 :: inplace_exp2
i_ln :: inplace_ln
i_log2 :: inplace_log2
i_log10 :: inplace_log10
i_log1p :: inplace_log1p
i_reciprocal :: inplace_reciprocal

i_sin :: inplace_sin
i_cos :: inplace_cos
i_tan :: inplace_tan
i_asin :: inplace_asin
i_acos :: inplace_acos
i_atan :: inplace_atan
i_degrees :: inplace_degrees
i_radians :: inplace_radians
i_degs :: inplace_degrees
i_rads :: inplace_radians
i_rad2deg :: inplace_degrees
i_deg2rad :: inplace_radians

i_sinh :: inplace_sinh
i_cosh :: inplace_cosh
i_tanh :: inplace_tanh
i_asinh :: inplace_asinh
i_acosh :: inplace_acosh
i_atanh :: inplace_atanh

i_conj :: inplace_conj
i_sinc :: inplace_sinc



o_sign :: outplace_sign
o_neg :: outplace_neg
o_abs :: outplace_abs

o_sq :: outplace_sq
o_sqrt :: outplace_sqrt
o_cbrt :: outplace_cbrt
o_exp :: outplace_exp
o_exp2 :: outplace_exp2
o_ln :: outplace_ln
o_log2 :: outplace_log2
o_log10 :: outplace_log10
o_log1p :: outplace_log1p
o_reciprocal :: outplace_reciprocal

o_sin :: outplace_sin
o_cos :: outplace_cos
o_tan :: outplace_tan
o_asin :: outplace_asin
o_acos :: outplace_acos
o_atan :: outplace_atan
o_degrees :: outplace_degrees
o_radians :: outplace_radians
o_degs :: outplace_degrees
o_rads :: outplace_radians
o_rad2deg :: outplace_degrees
o_deg2rad :: outplace_radians

o_sinh :: outplace_sinh
o_cosh :: outplace_cosh
o_tanh :: outplace_tanh
o_asinh ::outplace_asinh
o_acosh ::outplace_acosh
o_atanh ::outplace_atanh

o_conj :: outplace_conj
o_sinc :: outplace_sinc
