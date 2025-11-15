package mdarray

import "core:thread"
import "base:intrinsics"
import "core:math"
import "core:mem"
import "base:runtime"
import "../logging"

/*
Perform a custom element-wise binary operation on two different arrays.

Inputs:
- a: first multidimensional array.
- b: second multidimensional array.
- f: a binary procedure with arguments.
- args: arguments to be passed to the passed procedure.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.
- force_threaded(experimental): force this procedure to use threading, it will be ignored if the size of the arrays is smaller than the number of configured threads.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
element_wise_map :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	f: proc(_: T, _: T, _: ..$S) -> $R,
	args: ..S,
	allocator := context.allocator,
	location := #caller_location,
	force_threaded := false,
) -> (
	result: MdArray(R, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) ||
	intrinsics.type_is_boolean(T) {

	validate_initialized(a, location) or_return
	validate_initialized(b, location) or_return

	validate_shape_match(a, b, location=location) or_return
	result = make_mdarray(R, a.shape, allocator, location) or_return

	if size(a) >= THREAD_THR || (force_threaded && size(a) > N_THREADS) {
		_inner_element_wise_threaded(
			a, b, result,
			f, ..args,
			allocator=allocator, location=location
		)

		return result, true
	}
	else do for i in 0 ..< size(a) {
		result.buffer[i] = f(get_linear(a, i), get_linear(b, i), ..args)
	}
	return result, true
}


/*
Perform a custom element-wise binary operation on an arrays and a scalar.

Inputs:
- a: a multidimensional array.
- b: a scalar value.
- f: a binary procedure with arguments.
- args: arguments to be passed to the passed procedure.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.
- force_threaded(experimental): force this procedure to use threading, it will be ignored if the size of the arrays is smaller than the number of configured threads.

Returns:
- result: the resultant array.
- ok: an optional boolean for error handling.
*/
scalar_map :: proc(
	a: MdArray($T, $Nd),
	b: T,
	f: proc(_: T, _: T, _:..$S) -> $R,
	args: ..S,
	flip := false,
	allocator := context.allocator,
	location := #caller_location,
	force_threaded := false,
) -> (
	result: MdArray(R, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) ||
	intrinsics.type_is_boolean(T) {

	validate_initialized(a, location) or_return

	result = make_mdarray(R, a.shape, allocator, location) or_return

	if size(a) >= THREAD_THR || (force_threaded && size(a) > N_THREADS) {
		_inner_scalar_map_threaded(
			a, b, result,
			f, ..args, flip=flip,
			allocator=allocator, location=location
		)

		return result, true
	}

	if flip {
		for i in 0 ..< size(a) {
			result.buffer[i] = f(b, get_linear(a, i), ..args)
		}
		return result, true
	}

	for i in 0 ..< size(a) {
		result.buffer[i] = f(get_linear(a, i), b, ..args)
	}
	return result, true
}


@private
get_task_range :: proc(idx:int, s_len:int) -> (begin, end: int){
	n := s_len/(N_TASKS)

	begin = idx*n
	end =  (idx+1)*n

	if idx == N_TASKS-1{
		m := s_len%(N_TASKS)
		end += m
	}
	return begin, end
}


@private
_inner_element_wise_threaded :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	r: MdArray($R, Nd),
	f: proc(_: T, _: T, _: ..$S) -> R,
	args: ..S,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	ok: bool,
) where intrinsics.type_is_numeric(T) ||
	intrinsics.type_is_boolean(T) {

	Task_Data :: struct {
		a : ^MdArray(T, Nd),
		b : ^MdArray(T, Nd),
		r : ^MdArray(R, Nd),
		f: proc(T, T, ..S) -> R,
		args: []S,
		begin : int,
		end :int,
	}

	task_proc :: proc(t: thread.Task){
		d:= (^Task_Data)(t.data)

		for i in d.begin..<d.end{
			d.r.buffer[i] = d.f(get_linear(d.a^, i), get_linear(d.b^, i), ..d.args)
		}
	}

	a:=a
	b:=b
	r:=r
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
		task_data.b = &b
		task_data.r = &r
		task_data.f = f
		task_data.args = args
		task_data.begin, task_data.end = get_task_range(t_i, size(r))

		thread.pool_add_task(&pool, task_allocator, task_proc, task_data, t_i)
	}

	thread.pool_finish(&pool)
	
	return true
}


@private
_inner_scalar_map_threaded :: proc(
	a: MdArray($T, $Nd),
	b: T,
	r: MdArray($R, Nd),
	f: proc(_: T, _: T, _: ..$S) -> R,
	args: ..S,
	flip := false,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	ok: bool,
) where intrinsics.type_is_numeric(T) ||
	intrinsics.type_is_boolean(T) {

	Task_Data :: struct{
		a : ^MdArray(T, Nd),
		b : T,
		r : ^MdArray(R, Nd),
		flip: bool,
		f: proc(T, T, ..S) -> R,
		args: []S,
		begin : int,
		end :int,
	}

	task_proc :: proc(t: thread.Task){
		d:= (^Task_Data)(t.data)

		if d.flip {
			for i in d.begin..<d.end{
				d.r.buffer[i] = d.f(d.b, get_linear(d.a^, i), ..d.args)
			}
			return
		}
		for i in d.begin..<d.end{
			d.r.buffer[i] = d.f(get_linear(d.a^, i), d.b, ..d.args)
		}
	}

	a:=a
	b:=b
	r:=r

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
		task_data.b = b
		task_data.r = &r
		task_data.flip = flip
		task_data.f = f
		task_data.args = args
		task_data.begin, task_data.end = get_task_range(t_i, size(r))

		thread.pool_add_task(&pool, task_allocator, task_proc, task_data, t_i)
	}

	thread.pool_finish(&pool)
	
	return true
}


@(private = "file")
inner_add :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return a + b}
}
@(private = "file")
inner_subtract :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return a - b}
}
@(private = "file")
inner_mul :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return a * b}
}
@(private = "file")
inner_div :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return a / b}
}
@(private = "file")
inner_mod :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return a % b}
}
@(private = "file")
inner_remainder :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return a %% b}
}
@(private = "file")
inner_max :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.max(a, b)}
}
@(private = "file")
inner_min :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.min(a, b)}
}
@(private = "file")
inner_logaddexp :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.ln(math.exp(a) + math.exp(b))}
}
@(private = "file")
inner_gcd :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.gcd(a, b)}
}
@(private = "file")
inner_lcm :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.lcm(a, b)}
}
@(private = "file")
inner_hypot :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.sqrt(a * a, b * b)}
}
@(private = "file")
inner_atan2 :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {return math.atan2(a, b)}
}
@(private = "file")
inner_heaviside :: #force_inline proc($T: typeid) -> proc(_: T, _: T, _: ..T) -> T {
	return #force_inline proc(a: T, b: T, args: ..T) -> T {
			return a < 0 ? 0 : a > 0 ? 1 : b
		}
}


/*
Perform an element-wise addition operation on two different arrays.

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
add_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_add(T), allocator = allocator, location = location)
}


/*
Perform an element-wise subtraction operation on two different arrays.

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
subtract_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_subtract(T), allocator = allocator, location = location)
}


/*
Perform an element-wise multiplication operation on two different arrays.

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
mul_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_mul(T), allocator = allocator, location = location)
}


/*
Perform an element-wise division operation on two different arrays.

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
div_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_div(T), allocator = allocator, location = location)
}


/*
Perform an element-wise modulus operation on two different arrays.

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
mod_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_int(T) #optional_ok {
	return element_wise_map(a, b, inner_mod(T), allocator = allocator, location = location)
}


/*
Perform an element-wise minimum operation on two different arrays.

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
min_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_min(T), allocator = allocator, location = location)
}


/*
Perform an element-wise maximum operation on two different arrays.

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
max_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return element_wise_map(a, b, inner_max(T), allocator = allocator, location = location)
}


/*
Perform an element-wise logarithmic sum of exponentials operation on two different arrays.

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
logaddexp_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return element_wise_map(a, b, inner_logaddexp(T), allocator = allocator, location = location)
}


/*
Perform an element-wise greatest common denominator operation on two different arrays.

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
gcd_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_gcd(T), allocator = allocator, location = location)
}


/*
Perform an element-wise least common multiple operation on two different arrays.

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
lcm_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_lcm(T), allocator = allocator, location = location)
}


/*
Find the element-wise hypotenuse operation on two different arrays.

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
hypot_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_hypot(T), allocator = allocator, location = location)
}


/*
Perform an element-wise arctan2 operation on two different arrays.

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
atan2_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_atan2(T), allocator = allocator, location = location)
}


/*
Perform an element-wise Heaviside step operation on two different arrays.

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
heaviside_arrays :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return element_wise_map(a, b, inner_heaviside(T), allocator = allocator, location = location)
}


/*
Perform an element-wise addition operation on an arrays and a scalar.

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
add_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_add(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise subtraction operation on an arrays and a scalar.

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
subtract_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_subtract(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise multiplication operation on an arrays and a scalar.

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
mul_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_mul(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise division operation on an arrays and a scalar.

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
div_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_div(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise modulus operation on an arrays and a scalar.

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
mod_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_int(T) #optional_ok {
	return scalar_map(a, b, inner_mod(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise minimum operation on an arrays and a scalar.

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
min_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_min(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise maximum operation on an arrays and a scalar.

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
max_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(a, b, inner_max(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise logarithmic sum of exponentials operation on an arrays and a scalar.

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
logaddexp_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(a, b, inner_logaddexp(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise greatest common denominator operation on an arrays and a scalar.

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
gcd_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_gcd(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise least common multiple operation on an arrays and a scalar.

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
lcm_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_lcm(T), flip=false, allocator = allocator, location = location)
}


/*
Find the element-wise hypotenuse operation on an arrays and a scalar.

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
hypot_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_hypot(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise atan2 operation on an arrays and a scalar.

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
atan2_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_atan2(T), flip=false, allocator = allocator, location = location)
}


/*
Perform an element-wise Heaviside step operation on an arrays and a scalar.

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
heaviside_arrays_scalar :: proc(
	a: MdArray($T, $Nd),
	b: T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(a, b, inner_heaviside(T), flip=false, allocator = allocator, location = location)
}

/*
Perform an element-wise addition operation on a scalar and an array.

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
add_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_add(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise subtraction operation on a scalar and an array.

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
subtract_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_subtract(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise multiplication operation on a scalar and an array.

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
mul_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_mul(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise division operation on a scalar and an array.

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
div_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_div(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise modulus operation on a scalar and an array.

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
mod_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_int(T) #optional_ok {
	return scalar_map(b, a, inner_mod(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise minimum operation on a scalar and an array.

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
min_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_min(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise maximum operation on a scalar and an array.

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
max_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_max(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise logarithmic sum of exponentials operation on a scalar and an array.

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
logaddexp_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(b, a, inner_logaddexp(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise greatest common denominator operation on a scalar and an array.

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
gcd_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(b, a, inner_gcd(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise least common multiple operation on a scalar and an array.

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
lcm_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	return scalar_map(b, a, inner_lcm(T), flip=true, allocator = allocator, location = location)
}

/*
Find the element-wise hypotenuse operation on a scalar and an array.

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
hypot_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(b, a, inner_hypot(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise atan2 operation on a scalar and an array.

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
atan2_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return scalar_map(b, a, inner_atan2(T), flip=true, allocator = allocator, location = location)
}

/*
Perform an element-wise Heaviside step operation on a scalar and an array.

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
heaviside_scalar_array :: proc(
	a: $T,
	b: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return scalar_map(b, a, inner_heaviside(T), flip=true, allocator = allocator, location = location)
}


add :: proc {
	add_arrays,
	add_arrays_scalar,
	add_scalar_array,
}
subtract :: proc {
	subtract_arrays,
	subtract_arrays_scalar,
	subtract_scalar_array,
}
mul :: proc {
	mul_arrays,
	mul_arrays_scalar,
	mul_scalar_array,
}
div :: proc {
	div_arrays,
	div_arrays_scalar,
	div_scalar_array,
}
mod :: proc {
	mod_arrays,
	mod_arrays_scalar,
	mod_scalar_array,
}
minimum :: proc {
	min_arrays,
	min_arrays_scalar,
	min_scalar_array,
}
maximum :: proc {
	max_arrays,
	max_arrays_scalar,
	max_scalar_array,
}
logaddexp :: proc {
	logaddexp_arrays,
	logaddexp_arrays_scalar,
	logaddexp_scalar_array,
}
gcd :: proc {
	gcd_arrays,
	gcd_arrays_scalar,
	gcd_scalar_array,
}
lcm :: proc {
	lcm_arrays,
	lcm_arrays_scalar,
	lcm_scalar_array,
}
hypot :: proc {
	hypot_arrays,
	hypot_arrays_scalar,
	hypot_scalar_array,
}
atan2 :: proc {
	atan2_arrays,
	atan2_arrays_scalar,
	atan2_scalar_array,
}
heaviside :: proc {
	heaviside_arrays,
	heaviside_arrays_scalar,
	heaviside_scalar_array,
}
