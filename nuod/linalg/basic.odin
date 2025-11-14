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


/*
Produce a diagonal matrix whose diagonal elements are populated based 
on a one-dimensional array.

Inputs:
- mdarray: a one-dimenaional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the resultant diagonal matrix.
- ok: an optional boolean for error handling.
*/
make_diagonal_vector :: proc(	
	mdarray: md.MdArray($T, 1),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: md.MdArray(T, 2),
	ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	md.validate_initialized(mdarray, location=location) or_return

	n:= mdarray.shape[0]
	result = md.make_mdarray(T, [2]int{n, n}, allocator=allocator, location=location)

	for i in 0..<n{
		result.buffer[i*n+i] = md.get_linear(mdarray, i, location=location)
	}
	return result, true
}


make_diagonal_multidimensional :: proc(	
	$Nd: int, 
	mdarray: md.MdArray($T, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: md.MdArray(T, Nd+1),
	ok:bool
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	md.validate_initialized(mdarray, location=location) or_return

	n:= mdarray.shape[Nd-1]

	s_shape : [Nd+1]int
	s_shape[Nd] = n
	s_shape[Nd-1] = n
	for d in 0..<Nd-1{
		s_shape[d] = mdarray.shape[d]
	}

	result = md.make_mdarray(T, s_shape, allocator=allocator, location=location)

	nn := n*n
	for j in 0..<(md.size(mdarray)/n){
		jnn := j*nn
		for i in 0..<n{
			result.buffer[i*n+jnn+i] = md.get_linear(mdarray, i+j*n, location=location)
		}
	}
	return result, true
}


make_diagonal :: proc{
	make_diagonal_vector,
	make_diagonal_multidimensional
}


/*
Extract the diagonal elements of a matrix.

Inputs:
- mdarray: a matrix of two dimensions.
- offset: the offset from the main diagonal (may be negative).
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: a vector containing a copy of the diagonal elements.
- ok: an optional boolean for error handling.
*/
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

	min_dim := min(mdarray.shape[0], mdarray.shape[1])

	if abs(offset) > min_dim {
		logging.error(.ArguementError, "Recieved an offset larger than the dimensions of the matrix.")
		return
	}


	min_dim -= abs(offset)
	result = md.make_mdarray(T, [1]int{min_dim}, allocator=allocator, location=location)

	for i in 0..<min_dim{
		x:= offset<=0? i-offset : i
		y:= offset>=0? i+offset : i
		result.buffer[i] = md.get(mdarray, [2]int{x, y}, location=location) or_return
	}
	return result, true
}

/*
Extract the trace of a matrix.

Inputs:
- mdarray: a matrix of two dimensions.
- offset: the offset from the main diagonal (may be negative).
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the trace value.
- ok: an optional boolean for error handling.
*/
matrix_trace :: proc(	
	mdarray: md.MdArray($T, 2),
	offset:=0,
	location := #caller_location,
) -> (
	result: T,
	ok:bool
) where intrinsics.type_is_numeric(T) #optional_ok {
	
	md.validate_initialized(mdarray, location=location) or_return

	min_dim := min(mdarray.shape[0], mdarray.shape[1])

	if abs(offset) > min_dim {
		logging.error(.ArguementError, "Recieved an offset larger than the dimensions of the matrix.")
		return
	}

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

/*
Find the euclidean norm of vector. Treats multidimensional arrays as one vector.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the array.
- ok: an optional boolean for error handling.
*/
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

/*
Find the manhattan norm of vector. Treats multidimensional arrays as one vector.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the array.
- ok: an optional boolean for error handling.
*/
full_vector_manhattan_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.all_reduce_map(mdarray, inner_manhattan(T), cast(T)0, location=location)
}

/*
Find the chebyshev norm of vector. Treats multidimensional arrays as one vector.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the array.
- ok: an optional boolean for error handling.
*/
full_vector_chebyshev_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.all_reduce_map(mdarray, inner_chebyshev(T), cast(T)0, location=location)
}

/*
Find the L0 norm of vector. Treats multidimensional arrays as one vector.

Inputs:
- mdarray: a multidimensional array.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the array.
- ok: an optional boolean for error handling.
*/
full_vector_l0_norm :: proc(	
	mdarray: md.MdArray($T, $Nd),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.all_reduce_map(mdarray, inner_l0(T), cast(T)0, location=location)
}

/*
Find the norm of vector based on the selected norm type. Treats multidimensional 
arrays as one vector.

Inputs:
- mdarray: a multidimensional array.
- norm_type: the type of norm.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the array.
- ok: an optional boolean for error handling.
*/
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

/*
Find the euclidean norm of an array along a certain axis.

Inputs:
- Nd: number of dimensions of the array.
- mdarray: a multidimensional array.
- axis: the axis dimension along which the norm is computed.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the normed array, reduced in dimensions by one.
- ok: an optional boolean for error handling.
*/
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
	md.i_sqrt(norm_result, location=location) or_return 
	return norm_result, true
}

/*
Find the manhattan norm of an array along a certain axis.

Inputs:
- Nd: number of dimensions of the array.
- mdarray: a multidimensional array.
- axis: the axis dimension along which the norm is computed.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the normed array, reduced in dimensions by one.
- ok: an optional boolean for error handling.
*/
dim_vector_manhattan_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1),
	ok:bool,
) where intrinsics.type_is_float(T) #optional_ok {
	return md.dim_reduce_map(Nd, mdarray, axis, inner_manhattan(T), cast(T)0, allocator=allocator, location=location) 
}

/*
Find the chebyshev norm of an array along a certain axis.

Inputs:
- Nd: number of dimensions of the array.
- mdarray: a multidimensional array.
- axis: the axis dimension along which the norm is computed.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the normed array, reduced in dimensions by one.
- ok: an optional boolean for error handling.
*/
dim_vector_chebyshev_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.dim_reduce_map(Nd, mdarray, axis, inner_chebyshev(T), cast(T)0, allocator=allocator, location=location)
}

/*
Find the L0 norm of an array along a certain axis.

Inputs:
- Nd: number of dimensions of the array.
- mdarray: a multidimensional array.
- axis: the axis dimension along which the norm is computed.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the normed array, reduced in dimensions by one.
- ok: an optional boolean for error handling.
*/
dim_vector_l0_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	norm_result:md.MdArray(T, Nd-1), ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return md.dim_reduce_map(Nd, mdarray, axis, inner_l0(T), cast(T)0, allocator=allocator, location=location)
}

/*
Find the norm of an array along a certain axis based on the selected norm type.

Inputs:
- Nd: number of dimensions of the array.
- mdarray: a multidimensional array.
- axis: the axis dimension along which the norm is computed.
- norm_type: the type of norm to compute.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the normed array, reduced in dimensions by one.
- ok: an optional boolean for error handling.
*/
dim_vector_norm :: proc(
	$Nd :int,
	mdarray: md.MdArray($T, Nd),
	axis:int,
	norm_type : VectorNorm = .Euclidean,
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
	return
}

vector_norm :: proc{full_vector_norm, dim_vector_norm}


/*
Find the frobenius norm of a matrix.

Inputs:
- mdarray: a matrix.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
frobenius_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	return full_vector_euclidean_norm(mdarray, location=location)
}


/*
Find the nuclear norm of a matrix.

WARNING: This is not implemented yet.

Inputs:
- mdarray: a matrix.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
nuclear_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
	location := #caller_location,
) -> (
	norm_result:T,
	ok:bool
) where intrinsics.type_is_float(T) #optional_ok {
	logging.error(.NotImplemented, location=location)
	return 
}

/*
Find the spectral norm of a matrix.

WARNING: This is not implemented yet.

Inputs:
- mdarray: a matrix.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
spectral_matrix_norm :: proc(	
	mdarray: md.MdArray($T, 2),
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

/*
Find the infinity norm of a matrix.

Inputs:
- mdarray: a matrix.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
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

/*
Find the first norm of a matrix.

Inputs:
- mdarray: a matrix.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
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

/*
Find the negative infinity norm of a matrix.

Inputs:
- mdarray: a matrix.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
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

/*
Find the negative first norm of a matrix.

Inputs:
- mdarray: a matrix.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
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

/*
Find the norm of a matrix based on the selected matrix norm.

Inputs:
- mdarray: a matrix.
- norm_type: the type of norm.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the norm of the matrix.
- ok: an optional boolean for error handling.
*/
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
