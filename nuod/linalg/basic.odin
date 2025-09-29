package linalg

import "core:math"
import "core:fmt"
import "base:intrinsics"


import md "../mdarray"
import "../logging"


VectorNorm :: enum{
	L0,
	Zero,
	// manhattan
	L1,
	Manhattan,
	Absolute,

	// euclidean
	L2,
	Euclidean,

	// max norm
	Linfty,
	Chebyshev,
	Uniform,
	Max,
}


MatrixNorm :: enum{
	Frobenius,
	Nuclear,
	Spectral,
	Infty,
	NegInfty,
	First,
	NegFirst,
}


// Diagonal
matrix_diagonal :: proc(	
	mdarray: md.MdArray($T, 2),
	offset:=0,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: md.MdArray(T, 1),
	ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	md.validate_initialized(mdarray, location=location) or_return
	if abs(offset) > 2 {
		logging.error(.ArguementError, "Recieved an offset larger than the number of dimensions.")
		return
	}

	min_dim := min(mdarray.shape[0], mdarray.shape[1])

	min_dim -= abs(offset)
	result = md.make_mdarray(T, [1]int{min_dim}, allocator=allocator, location=location)

	for i in 0..<min_dim{
		x:= offset<=0? i-offset : i
		y:= offset>=0? i+offset : i
		result.buffer[i] = md.get(mdarray, [2]int{x, y}, location=location) or_return
	}
	return result, true
}

// Trace
matrix_trace :: proc(	
	mdarray: md.MdArray($T, 2),
	offset:=0,
	location := #caller_location,
) -> (
	result: T,
	ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	
	md.validate_initialized(mdarray, location=location) or_return
	if abs(offset) > 2 {
		logging.error(.ArguementError, "Recieved an offset larger than the number of dimensions.")
		return
	}

	min_dim := min(mdarray.shape[0], mdarray.shape[1])
	for i in 0..<(min_dim-abs(offset)){
		x:= offset<=0? i-offset : i
		y:= offset>=0? i+offset : i
		result += md.get(mdarray, [2]int{x, y}, location=location) or_return
	}
	return result, true
}


// vector norm

@(private="file")
inner_euclidean :: #force_inline  proc($T: typeid)-> proc(T, T, ..T)->T {
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return accum + val*val}
}
@(private="file")
inner_manhattan :: #force_inline  proc($T: typeid)-> proc(T, T, ..T)->T {
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return accum + abs(val)}
}
@(private="file")
inner_chebyshev :: #force_inline  proc($T: typeid)-> proc(T, T, ..T)->T {
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return max(accum, abs(val))}
}
@(private="file")
inner_l0 :: #force_inline  proc($T: typeid)-> proc(T, T, ..T)->T {
	return #force_inline proc (accum: T, val: T, args: ..T) -> T { return accum +  (val==0? 0: 1) }
}


full_vector_euclidean_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	norm_result = md.all_reduce_map(mdarray, inner_euclidean(T), cast(T)0, location=location) or_return
	norm_result = math.sqrt(norm_result)
	return norm_result, true
}


full_vector_manhattan_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.all_reduce_map(mdarray, inner_manhattan(T), cast(T)0, location=location)
}


full_vector_chebyshev_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.all_reduce_map(mdarray, inner_chebyshev(T), cast(T)0, location=location)
}


full_vector_l0_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.all_reduce_map(mdarray, inner_l0(T), cast(T)0, location=location)
}


full_vector_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	norm_type : VectorNorm = .Euclidean,
	location := #caller_location,
) -> (
	accum:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	switch norm_type {
		case .L0, .Zero:			
			return full_vector_l0_norm(mdarray, location=location)
		case .L1, .Manhattan, .Absolute:
			return full_vector_manhattan_norm(mdarray, location=location)
		case .L2, .Euclidean:
			return full_vector_euclidean_norm(mdarray, location=location)
		case .Linfty, .Chebyshev, .Max, .Uniform:
			return full_vector_chebyshev_norm(mdarray, location=location)
	}
	return
}


dim_vector_euclidean_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	norm_result = md.dim_reduce_map(Nd, mdarray, axis, inner_euclidean(T), cast(T)0, allocator=allocator, location=location) or_return
	norm_result = md.i_sqrt(norm_result, location=location) or_return 
	return norm_result, true
}


dim_vector_manhattan_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.dim_reduce_map(Nd, mdarray, axis, inner_manhattan(T), cast(T)0, allocator=allocator, location=location) or_return
}


dim_vector_chebyshev_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.dim_reduce_map(Nd, mdarray, axis, inner_chebyshev(T), cast(T)0, allocator=allocator, location=location) or_return
}

dim_vector_l0_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.dim_reduce_map(Nd, mdarray, axis, inner_l0(T), cast(T)0, allocator=allocator, location=location) or_return
}

dim_vector_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	norm : VectorNorm = .Euclidean,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	switch norm_type {
		case .L0, .Zero:
			return dim_vector_l0_norm(Nd, mdarray, axis, allocator,  location)
		case .L1, .Manhattan, .Absolute:
			return dim_vector_manhattan_norm(Nd, mdarray, axis, allocator,  location)
		case .L2, .Euclidean:
			return dim_vector_euclidean_norm(Nd, mdarray, axis, allocator,  location)
		case .Linfty, .Chebyshev, .Max, .Uniform:
			return dim_vector_chebyshev_norm(Nd, mdarray, axis, allocator,  location)
	}
}

vector_norm :: proc{full_vector_norm, dim_vector_norm}


// matrix norm
frobenius_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return full_vector_euclidean_norm(mdarray, location=location)
}


nuclear_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	logging.error(.NotImplemented, location=location)
	return 
}


spectral_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	logging.error(.NotImplemented, location=location)
	return 
}


@(private="file")
infty_first_matrix_norm_selector :: proc(	
	mdarray: md.MdArray($T, 2),
	axis:=1,
	max_reduce:= true,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	abs_arr := md.o_abs(mdarray, allocator=allocator, location=location) or_return
	defer md.free_mdarray(abs_arr)
	sum_dim := md.dim_reduce_sum(2, abs_arr, axis, T(0), allocator=allocator, location=location) or_return
	defer md.free_mdarray(sum_dim)
	if max_reduce{
		return md.all_reduce_max(sum_dim, location=location)
	} else {
		return md.all_reduce_min(sum_dim, location=location)
	}
}


infty_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return infty_first_matrix_norm_selector(
		mdarray,
		axis=1,
		max_reduce=true,
		allocator=allocator,
		location=location
	)
}


first_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return infty_first_matrix_norm_selector(
		mdarray,
		axis=0,
		max_reduce=true,
		allocator=allocator,
		location=location
	)
}


neg_infty_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return infty_first_matrix_norm_selector(
		mdarray,
		axis=1,
		max_reduce=false,
		allocator=allocator,
		location=location
	)
}


neg_first_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return infty_first_matrix_norm_selector(
		mdarray,
		axis=0,
		max_reduce=false,
		allocator=allocator,
		location=location
	)
}


matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	norm_type:MatrixNorm = MatrixNorm.Frobenius,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	switch norm_type{
		case .Frobenius:
			return frobenius_matrix_norm(mdarray, location=location)
		case .Nuclear:
			return nuclear_matrix_norm(mdarray, allocator=allocator, location=location)
		case .Spectral:
			return spectral_matrix_norm(mdarray, allocator=allocator, location=location)
		case .First:
			return first_matrix_norm(mdarray, allocator=allocator, location=location)
		case .NegFirst:
			return neg_first_matrix_norm(mdarray, allocator=allocator, location=location)
		case .Infty:
			return infty_matrix_norm(mdarray, allocator=allocator, location=location)
		case .NegInfty:
			return neg_infty_matrix_norm(mdarray, allocator=allocator, location=location)
	}
	return
}
