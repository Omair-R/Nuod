package random

import "core:math/rand"
import "base:intrinsics"

import md "../mdarray"
import "../logging"

/*
Generate a multidimensional array and fill it with random values sampled from an
exponential distribution.

NOTE: the type of 'lambda' decides the type of generated array.

Inputs:
- lambda: a control parameter for the distribution.
- shape: the shape of generated array.
- gen: the random number generator object.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: the generated array.
- ok: an optional boolean for error handling.
*/
exponential_sample :: proc(
	lambda: $T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {

	when T == f32{
		return one_arg_wrapper(lambda, shape, rand.float32_exponential, gen, allocator, location)
	} else when T == f64 {
		return one_arg_wrapper(lambda, shape, rand.float64_exponential, gen, allocator, location)
	} else{
			logging.error(.ArguementError, "Provided type is unsupported.", location)
	}
}

/*
Generate a multidimensional array and fill it with random values sampled from a
normal distribution.

NOTE: the type of 'lambda' decides the type of generated array.

Inputs:
- mean: a control parameter for the distribution.
- stddev: a control parameter for the distribution.
- shape: the shape of generated array.
- gen: the random number generator object.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: the generated array.
- ok: an optional boolean for error handling.
*/
normal_sample :: proc(
	mean: $T,
	stddev: T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {

	when T == f32{
		return two_arg_wrapper(mean, stddev, shape, rand.float32_normal, gen, allocator, location)
	} else when T == f64 {
		return two_arg_wrapper(mean, stddev, shape, rand.float64_normal, gen, allocator, location)
	} else{
			logging.error(.ArguementError, "Provided type is unsupported.", location)
	}
}

/*
Generate a multidimensional array and fill it with random values sampled from a
laplace distribution.

NOTE: the type of 'lambda' decides the type of generated array.

Inputs:
- mean: a control parameter for the distribution.
- b: a control parameter for the distribution.
- shape: the shape of generated array.
- gen: the random number generator object.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: the generated array.
- ok: an optional boolean for error handling.
*/
laplace_sample :: proc(
	mean: $T,
	b: T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {

	when T == f32{
		return two_arg_wrapper(mean, b, shape, rand.float32_laplace, gen, allocator, location)
	} else when T == f64 {
		return two_arg_wrapper(mean, b, shape, rand.float64_laplace, gen, allocator, location)
	} else{
			logging.error(.ArguementError, "Provided type is unsupported.", location)
	}
}

/*
Generate a multidimensional array and fill it with random values sampled from a
gamma distribution.

NOTE: the type of 'lambda' decides the type of generated array.

Inputs:
- alpha: a control parameter for the distribution.
- beta: a control parameter for the distribution.
- shape: the shape of generated array.
- gen: the random number generator object.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: the generated array.
- ok: an optional boolean for error handling.
*/
gamma_sample :: proc(
	alpha: $T,
	beta: T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {

	when T == f32{
		return two_arg_wrapper(alpha, beta, shape, rand.float32_gamma, gen, allocator, location)
	} else when T == f64 {
		return two_arg_wrapper(alpha, beta, shape, rand.float64_gamma, gen, allocator, location)
	} else{
			logging.error(.ArguementError, "Provided type is unsupported.", location)
	}
}

/*
Generate a multidimensional array and fill it with random values sampled from a
beta distribution.

NOTE: the type of 'lambda' decides the type of generated array.

Inputs:
- alpha: a control parameter for the distribution.
- beta: a control parameter for the distribution.
- shape: the shape of generated array.
- gen: the random number generator object.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: the generated array.
- ok: an optional boolean for error handling.
*/
beta_sample :: proc(
	alpha: $T,
	beta: T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {

	when T == f32{
		return two_arg_wrapper(alpha, beta, shape, rand.float32_beta, gen, allocator, location)
	} else when T == f64 {
		return two_arg_wrapper(alpha, beta, shape, rand.float64_beta, gen, allocator, location)
	} else{
			logging.error(.ArguementError, "Provided type is unsupported.", location)
	}
}


