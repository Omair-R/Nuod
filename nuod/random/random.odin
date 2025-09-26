package random

import "core:math/rand"
import "base:intrinsics"

import md "../mdarray"
import "../logging"


@private
no_arg_wrapper :: proc(
	$T: typeid,
	shape: [$Nd]int,
	f : proc(rand.Generator)->T,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	md.validate_positive_shape(shape, location=location) or_return
	
	mdarray = md.make_mdarray(T, shape, allocator) or_return

	for i in 0 ..< len(mdarray.buffer) {
		mdarray.buffer[i] = f(gen)
	}

	return mdarray, true
}


@private
one_arg_wrapper :: proc(
	first: $T,
	shape: [$Nd]int,
	f: proc(T, rand.Generator) -> T,
	gen:=context.random_generator,
	allocator:= context.allocator,
	location:= #caller_location,
)-> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
){
	md.validate_positive_shape(shape, location=location) or_return
	mdarray = md.make_mdarray(T, shape, allocator) or_return

	for i in 0 ..< len(mdarray.buffer) {
		mdarray.buffer[i] = f(first, gen)
	}

	return mdarray, true
}


@private
two_arg_wrapper :: proc(
	first: $T,
	second: T,
	shape: [$Nd]int,
	f: proc(T, T, rand.Generator) -> T,
	gen:=context.random_generator,
	allocator:= context.allocator,
	location:= #caller_location,
)-> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
){
	md.validate_positive_shape(shape, location=location) or_return
	mdarray = md.make_mdarray(T, shape, allocator) or_return

	for i in 0 ..< len(mdarray.buffer) {
		mdarray.buffer[i] = f(first, second, gen)
	}

	return mdarray, true
}


@private
casted_one_arg_wrapper :: proc(
	first: $T,
	f : proc(T, rand.Generator)->T,
	$CastType : typeid,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(CastType, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	md.validate_positive_shape(shape, location=location) or_return
	mdarray = md.make_mdarray(CastType, shape, allocator) or_return


	for i in 0 ..< len(mdarray.buffer) {
		mdarray.buffer[i] = cast(CastType)f(first, gen)
	}

	return mdarray, true
}


random_int :: proc(
	$T : typeid,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	when T == int {
		return one_arg_wrapper(max(int), shape, rand.int_max, gen, allocator, location)
	} else when T == i32{
		return no_arg_wrapper(i32, shape, rand.int31, gen, allocator, location)
	} else when T == i64{
		return no_arg_wrapper(i64, shape, rand.int63, gen, allocator, location)
	} else when T == i128{
		return no_arg_wrapper(i128,shape, rand.int127, gen, allocator, location)
	} else when T == u32{
		return no_arg_wrapper(u32, shape, rand.uint32, gen, allocator, location)
	} else when T == u64{
		return no_arg_wrapper(u64, shape, rand.uint64, gen, allocator, location)
	} else when T == u128{
		return no_arg_wrapper(u128,shape, rand.uint128, gen, allocator, location)
	} else {		
		logging.error(.ArguementError, "Provided type is unsupported.", location)
		return
	}
}


random_int_max :: proc(
	max_value: $T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_integer(T) #optional_ok {
	when T == int {
		return one_arg_wrapper(max_value, shape, rand.int_max, gen, allocator, location)
	} else when T == i32{
		return one_arg_wrapper(max_value, shape, rand.int31_max, gen, allocator, location)
	} else when T == i64{
		return one_arg_wrapper(max_value, shape, rand.int63_max, gen, allocator, location)
	} else when T == i128{
		return one_arg_wrapper(max_value, shape, rand.int127_max, gen, allocator, location)
	} else when T == u32{
		return casted_one_arg_wrapper(cast(i32)max_value, rand.int31_max, u32, shape, gen, allocator, location)
	} else when T == u64{
		return casted_one_arg_wrapper(cast(i64)max_value, rand.int63_max, u64, shape, gen, allocator, location)
	} else when T == u128{
		return casted_one_arg_wrapper(cast(i128)max_value, rand.int127_max, u128, shape, gen, allocator, location)
	} else {		
		logging.error(.ArguementError, "Provided type is unsupported.", location)
		return
	}
}


random_float :: proc(
	$T : typeid,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	when T == f32{
		return no_arg_wrapper(f32, shape, rand.float32, gen, allocator, location)
	} else when T == f64 {
		return no_arg_wrapper(f64, shape, rand.float64, gen, allocator, location)
	} else{
		logging.error(.ArguementError, "Provided type is unsupported.", location)
		return
	}
}

random_float_range :: proc(
	low: $T,
	high: T,
	shape: [$Nd]int,
	gen:=context.random_generator,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	when T == f32{
		return two_arg_wrapper(low, high, shape, rand.float32_range, gen, allocator, location)
	} else when T == f64 {
		return two_arg_wrapper(low, high, shape, rand.float64_range, gen, allocator, location)
	} else{
		logging.error(.ArguementError, "Provided type is unsupported.", location)
		return
	}
}

