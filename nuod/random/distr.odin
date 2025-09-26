package random

import "core:math/rand"
import "base:intrinsics"

import md "../mdarray"
import "../logging"


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


