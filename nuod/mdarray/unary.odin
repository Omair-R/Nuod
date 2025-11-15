package mdarray

import "base:intrinsics"
import "core:math"
import "core:mem"
import "base:runtime"
import "core:thread"

import "../logging"
/*
Apply a custom unary operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- f: a unary procedure.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_unary_map :: proc(
	mdarray: MdArray($T, $Nd),
	f: proc(^T),
	allocator:= context.allocator,
	location := #caller_location,
	force_threaded := false,
) -> (
	 result:MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {
	
	result = copy_array(mdarray, allocator=allocator, location=location) or_return

	if size(mdarray) >= THREAD_THR || (force_threaded && size(mdarray) > N_THREADS) {
		_inner_unary_map_threaded(
			result, f,
			allocator, location 
		)
		return result, true
	}
	for i in 0..<size(mdarray){		
		f(get_linear_ref(result, i))
	}

	return result, true
}

	
/*
Apply a custom unary operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- f: a unary procedure.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_unary_map :: proc(
	mdarray: MdArray($T, $Nd),
	f: proc(^T), 	
	location := #caller_location,
	force_threaded := false,
) -> (
	 ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T){
	
	validate_initialized(mdarray, location) or_return
	
	if size(mdarray) >= THREAD_THR || (force_threaded && size(mdarray) > N_THREADS) {
		_inner_unary_map_threaded(
			mdarray, f,
			context.allocator, location //the context allocator is temporary 
		)
		return true
	}

	for i in 0..<size(mdarray){		
		f(get_linear_ref(mdarray, i))
	}
	
	return true
}

@private
_inner_unary_map_threaded :: proc(
	a: MdArray($T, $Nd),
	f: proc(^T), 	
	allocator := context.allocator,
	location := #caller_location,
) -> (
	ok: bool,
) where intrinsics.type_is_numeric(T) ||
	intrinsics.type_is_boolean(T) {

	Task_Data :: struct{
		a : ^MdArray(T, Nd),
		f: proc(^T), 	
		begin : int,
		end :int,
	}

	task_proc :: proc(t: thread.Task){
		d:= (^Task_Data)(t.data)

		for i in d.begin..<d.end{
			d.f(get_linear_ref(d.a^, i))
		}
	}

	a:=a

	pool : thread.Pool
	thread.pool_init(&pool, allocator, N_THREADS)
	thread.pool_start(&pool)

	defer thread.pool_destroy(&pool)

	task_data_a : [N_TASKS]Task_Data

	for t_i in 0..<(N_TASKS){
		task_allocator : mem.Allocator
		task_allocator = runtime.nil_allocator()

		task_data := &task_data_a[t_i]

		task_data.a = &a
		task_data.f = f
		task_data.begin, task_data.end = get_task_range(t_i, size(a))

		thread.pool_add_task(&pool, task_allocator, task_proc, task_data, t_i)
	}

	thread.pool_finish(&pool)
	
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
inner_reciprocal :: #force_inline proc($T: typeid)-> (proc(^T)) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
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


/*
Apply the sign operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_sign :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_sign(T), allocator, location)
}

/*
Apply the sign operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_sign :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_sign(T), location)
}


/*
Apply the negating operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_neg :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_neg(T), allocator, location)
}

/*
Apply the negating operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_neg :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_neg(T), location)
}


/*
Apply the absolute operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_abs :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_abs(T), allocator, location)
}

/*
Apply the absolute operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_abs :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_abs(T), location)
}

// ----

/*
Apply the square operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_sq :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_sq(T), allocator, location)
}

/*
Apply the square operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_sq :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_sq(T), location)
}


/*
Apply the square root operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_sqrt :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_sqrt(T), allocator, location)
}

/*
Apply the square root operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_sqrt :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sqrt(T), location)
}


/*
Apply the cube root operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_cbrt :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_cbrt(T), allocator, location)
}

/*
Apply the cube root operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_cbrt :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_cbrt(T), location)
}


/*
Apply the exponential operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_exp :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_exp(T), allocator, location)
}

/*
Apply the exponential operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_exp :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_exp(T), location)
}


/*
Apply the exponential operator (base of 2) to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_exp2 :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_exp2(T), allocator, location)
}

/*
Apply the exponential operator (base of 2) to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_exp2 :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_exp2(T), location)
}


/*
Apply the natural logarithm operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_ln :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_ln(T), allocator, location)
}

/*
Apply the natural logarithm operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_ln :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_ln(T), location)
}


/*
Apply the logarithm operator (base of 2) to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_log2 :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_log2(T), allocator, location)
}

/*
Apply the logarithm operator (base of 2) to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_log2 :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_log2(T), location)
}


/*
Apply the logarithm operator (base of 10) to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_log10 :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_log10(T), allocator, location)
}

/*
Apply the logarithm operator (base of 10) to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_log10 :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_log10(T), location)
}


/*
Apply the natural logarithm operator with one plus to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_log1p :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_log1p(T), allocator, location)
}

/*
Apply the natural logarithm operator with one plus to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_log1p :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_log1p(T), location)
}


/*
Apply the reciprocal operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_reciprocal :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_reciprocal(T), allocator, location)
}

/*
Apply the reciprocal operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_reciprocal :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_numeric(T) {
	return inplace_unary_map(mdarray, inner_reciprocal(T), location)
}


// -----

/*
Apply the sine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_sin :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_sin(T), allocator, location)
}

/*
Apply the sine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_sin :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sin(T), location)
}


/*
Apply the cosine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_cos :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_cos(T), allocator, location)
}

/*
Apply the cosine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_cos :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_cos(T), location)
}


/*
Apply the tan operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_tan :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_tan(T), allocator, location)
}

/*
Apply the tan operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_tan :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_tan(T), location)
}


/*
Apply the arcsine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_asin :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_asin(T), allocator, location)
}

/*
Apply the arcsine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_asin :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_asin(T), location)
}


/*
Apply the arccosine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_acos :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_acos(T), allocator, location)
}

/*
Apply the arccosine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_acos :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_acos(T), location)
}


/*
Apply the arctan operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_atan :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_atan(T), allocator, location)
}

/*
Apply the arctan operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_atan :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_atan(T), location)
}


/*
Convert all elements in the array from radian to degress out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_degrees :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_degrees(T), allocator, location)
}

/*
Convert all elements in the array from radian to degress in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_degrees :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_degrees(T), location)
}


/*
Convert all elements in the array from degrees to radian out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_radians :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_radians(T), allocator, location)
}

/*
Convert all elements in the array from degrees to radian in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_radians :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_radians(T), location)
}


// ------

/*
Apply the hyperpolic sine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_sinh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_sinh(T), allocator, location)
}

/*
Apply the hyperpolic sine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_sinh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_sinh(T), location)
}


/*
Apply the hyperpolic cosine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_cosh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_cosh(T), allocator, location)
}

/*
Apply the hyperpolic cosine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_cosh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_cosh(T), location)
}


/*
Apply the hyperpolic tan operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_tanh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_tanh(T), allocator, location)
}

/*
Apply the hyperpolic tan operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_tanh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_tanh(T), location)
}


/*
Apply the hyperpolic arcsine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_asinh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_asinh(T), allocator, location)
}

/*
Apply the hyperpolic arcsine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_asinh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_asinh(T), location)
}


/*
Apply the hyperpolic arccosine operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_acosh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_acosh(T), allocator, location)
}

/*
Apply the hyperpolic arccosine operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_acosh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_acosh(T), location)
}


/*
Apply the hyperpolic arctan operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_atanh :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_atanh(T), allocator, location)
}

/*
Apply the hyperpolic arctan operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_atanh :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_float(T) {
	return inplace_unary_map(mdarray, inner_atanh(T), location)
}


// ------

/*
Apply the conjugate operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_conj :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_complex(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_conj(T), allocator, location)
}

/*
Apply the conjugate operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
inplace_conj :: proc(
	mdarray : MdArray($T, $Nd),
	location := #caller_location,
) -> (
	ok: bool
) where intrinsics.type_is_complex(T) {
	return inplace_unary_map(mdarray, inner_conj(T), location)
}


/*
Apply the sine cardinal operator to all elements in the array out of place.

Inputs:
- mdarray: a multidimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
outplace_sinc :: proc(
	mdarray : MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok: bool
) where intrinsics.type_is_float(T) #optional_ok {
	return outplace_unary_map(mdarray, inner_sinc(T), allocator, location)
}

/*
Apply the sine cardinal operator to all elements in the array in place.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- ok: an optional boolean for error handling.
*/
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
