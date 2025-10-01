package mdarray

import "core:fmt"
import "base:intrinsics"
import "core:math"

import "../logging"

all_reduce_map :: proc(
	mdarray: MdArray($T, $Nd),
	f: proc(T, T, ..T) -> T,
	initial:T,
	args: ..T,
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location=location) or_return

	accum = initial
	for i in 0..<size(mdarray){
		accum = f(accum, get_linear(mdarray, i), ..args)
	}
	return accum, true
}

// First args is always the size of the of the axis dim.
dim_reduce_map :: proc(
	$Nd: int,
	mdarray: MdArray($T, Nd),
	axis:int,
	f: proc(T, T, ..T) -> T,
	initial:T,
	args: ..T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd-1),
	ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {
	
	validate_initialized(mdarray, location=location) or_return

	result_shape : [Nd-1]int
	offset:= 0
	for d in 0..<len(mdarray.shape) {
		if d == axis {
			offset += 1
			continue
		}
		result_shape[d-offset] = mdarray.shape[d]
	}

	result = make_mdarray(T, result_shape, allocator=allocator, location=location) or_return

	nd_idx:[Nd]int
	reduced_idx:[Nd-1]int
	bf_idx:int
	for i in 0..<size(result){
		reduced_idx = from_buffer_index(result, i, location=location) or_return

		offset = 0
		for d in 0..<len(mdarray.shape){
			if d == axis {
				offset += 1
				continue
			}
			nd_idx[d] = reduced_idx[d-offset]
		}

		result.buffer[i] = initial
		for j in 0..<(mdarray.shape[axis]){
			nd_idx[axis] = j
			bf_idx = to_buffer_index(mdarray, nd_idx, location=location) or_return
			result.buffer[i] = f(result.buffer[i], mdarray.buffer[bf_idx], ..args)
		}
	}

	return result, true
}


keepdim_reduce_map :: proc(
	$Nd: int,
	mdarray: MdArray($T, Nd),
	axis:int,
	f: proc(T, T, ..T) -> T,
	initial:T,
	args: ..T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {
	
	validate_initialized(mdarray, location=location) or_return

	result_shape := mdarray.shape
	result_shape[axis] = 1

	result = make_mdarray(T, result_shape, allocator=allocator, location=location) or_return

	nd_idx:[Nd]int
	reduced_idx:[Nd]int
	bf_idx:int
	for i in 0..<size(result){
		reduced_idx = from_buffer_index(result, i, location=location) or_return
		result.buffer[i] = initial
		for j in 0..<(mdarray.shape[axis]){
			reduced_idx[axis] = j
			bf_idx = to_buffer_index(mdarray, reduced_idx, location=location) or_return
			result.buffer[i] = f(result.buffer[i], mdarray.buffer[bf_idx], ..args)
		}
	}

	return result, true
}


@(private="file")
inner_sum :: #force_inline  proc($T: typeid)-> proc(T, T,..T)->T{
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return accum + val}
}
@(private="file")
inner_prod :: #force_inline proc($T: typeid)-> proc(T, T,..T)->T{
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return accum * val}
}
@(private="file")
inner_max :: #force_inline proc($T: typeid)-> proc(T, T,..T)->T{
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return math.max(accum, val)}
}
@(private="file")
inner_min :: #force_inline proc($T: typeid)-> proc(T, T,..T)->T{
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return math.min(accum, val)}
}
@(private="file")
inner_avg :: #force_inline proc($T: typeid)-> proc(T, T,..T)->T{
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return accum + (val/args[0])}
}

all_reduce_sum_no_init :: proc(	
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return all_reduce_map(mdarray, inner_sum(T), cast(T)0, location=location)
}

all_reduce_sum_with_init :: proc(	
	mdarray: MdArray($T, $Nd),
	initial:T,
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return all_reduce_map(mdarray, inner_sum(T), cast(T)initial, location=location)
}

all_reduce_sum :: proc{all_reduce_sum_no_init, all_reduce_sum_with_init}

all_reduce_prod_no_init :: proc(	
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return all_reduce_map(mdarray, inner_prod(T), cast(T)1, location=location)
}


all_reduce_prod_with_init :: proc(	
	mdarray: MdArray($T, $Nd),
	initial:T,
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return all_reduce_map(mdarray, inner_prod(T), cast(T)initial, location=location)
}


all_reduce_prod :: proc{all_reduce_prod_no_init, all_reduce_prod_with_init}


all_reduce_min :: proc(	
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	initial := max(T)
	return all_reduce_map(mdarray, inner_min(T), initial, location=location)
}


all_reduce_max :: proc(	
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	initial := min(T)
	return all_reduce_map(mdarray, inner_max(T), initial, location=location)
}


all_reduce_avg :: proc(	
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	accum:T, ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	accum, ok = all_reduce_map(mdarray, inner_sum(T), cast(T)0, location=location)
	return accum/size(mdarray), ok
}


dim_reduce_sum_no_init :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return dim_reduce_map(Nd, mdarray, axis, inner_sum(T), cast(T)0, allocator=allocator, location=location)
}


dim_reduce_sum_with_init :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	initial:T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return dim_reduce_map(Nd, mdarray, axis, inner_sum(T), cast(T)initial, allocator=allocator, location=location)
}

dim_reduce_sum :: proc{dim_reduce_sum_no_init, dim_reduce_sum_with_init}


dim_reduce_prod_no_init :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return dim_reduce_map(Nd, mdarray, axis, inner_prod(T), cast(T)1, allocator=allocator, location=location)
}


dim_reduce_prod_with_init :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	initial:T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return dim_reduce_map(Nd, mdarray, axis, inner_prod(T), cast(T)initial, allocator=allocator, location=location)
}

dim_reduce_prod :: proc{dim_reduce_prod_no_init, dim_reduce_prod_with_init}


dim_reduce_min :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	initial := max(T)
	return dim_reduce_map(Nd, mdarray, axis, inner_min(T), initial, allocator=allocator, location=location)
}


dim_reduce_max :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	initial := min(T)
	return dim_reduce_map(Nd, mdarray, axis, inner_max(T), initial, allocator=allocator, location=location)
}


dim_reduce_avg :: proc(
	$Nd :int,
	mdarray: MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return dim_reduce_map(Nd, mdarray, axis, inner_avg(T), cast(T)0, cast(T)mdarray.shape[axis], allocator=allocator, location=location)
}


keepdim_reduce_sum_no_init :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return keepdim_reduce_map(Nd, mdarray, axis, inner_sum(T), cast(T)0, allocator=allocator, location=location)
}


keepdim_reduce_sum_with_init :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	initial:T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return keepdim_reduce_map(Nd, mdarray, axis, inner_sum(T), cast(T)initial, allocator=allocator, location=location)
}

keepdim_reduce_sum :: proc{keepdim_reduce_sum_no_init, keepdim_reduce_sum_with_init}


keepdim_reduce_prod_no_init :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return keepdim_reduce_map(Nd, mdarray, axis, inner_prod(T), cast(T)1, allocator=allocator, location=location)
}


keepdim_reduce_prod_with_init :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	initial:T,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return dim_reduce_map(Nd, mdarray, axis, inner_prod(T), cast(T)initial, allocator=allocator, location=location)
}

keepdim_reduce_prod :: proc{keepdim_reduce_prod_no_init, keepdim_reduce_prod_with_init}



keepdim_reduce_min :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	initial := max(T)
	return keepdim_reduce_map(Nd, mdarray, axis, inner_min(T), initial, allocator=allocator, location=location)
}


keepdim_reduce_max :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	initial := min(T)
	return keepdim_reduce_map(Nd, mdarray, axis, inner_max(T), initial, allocator=allocator, location=location)
}


keepdim_reduce_avg :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	accum:MdArray(T, Nd), ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	return keepdim_reduce_map(Nd, mdarray, axis, inner_avg(T), cast(T)0, cast(T)mdarray.shape[axis], allocator=allocator, location=location)
}
