package mdarray

import errors "../errors"
import logging "../logging"
import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:slice"


/*
A structure that represents a multi-dimensional array. it stores all data in a singular
slice. The dimensionality of the array is decided based on the shape field. To traverse
the array an array of strides is precomputed based on the shape of the array.

This structure does not now allows own the data in its buffer. If that's the case, then
it's is called a view. Views may have shapes and strides that don't corresbond to the
buffer length, typically because the view narrows the array or the shape of the arry
has been permutated. How these fields are managed should not concern the user, as they
are handled internally by Nuod.

CAUTION: do not create instances of this structure manually. Used the make_mdarray
procedure instead. It ensures that the array is valid and can be traversed correctly.
*/
MdArray :: struct($T: typeid, $Nd: int) {
	buffer:  []T,
	shape:   [Nd]int,
	strides: [Nd]int,

	// View specific variables
	is_view: bool,
	offset: int,
	shape_strides: [Nd]int,
}


/*
Retrives the number of dimensions of the array.

Inputs:
- mdarray: a multidimsional array of any internal type.

Returns:
- number of dimensions.
*/
ndim :: proc(mdarray: MdArray($T, $Nd)) -> int {
	return Nd
}


/*
Retrives the total number of elements in the array. If the array is a view,
it will only retrive the number of viewed elements only.

Inputs:
- mdarray: a multidimsional array of any internal type.

Returns:
- number of elements.
*/
size :: proc(mdarray: MdArray($T, $Nd)) -> int {
	if mdarray.is_view {
		size:= 1
		for dim in mdarray.shape{
			size *=dim
		}
		return size
	}
	return len(mdarray.buffer)
}


/*
Retrives the internal slice of the array. It does not care if the array is a view
it will retrieve the original buffer. Do not abuse it. If you need a flattened view
or copy of an array use the flatten_view or flatten_copy procedures, respectively.

Inputs:
- mdarray: a multidimsional array of any internal type.

Returns:
- the original slice buffer.
*/

ravel :: proc(mdarray: MdArray($T, $Nd)) -> []T {
	return mdarray.buffer[:]
}


/*
Retrives the internal type of the array.

Inputs:
- mdarray: a multidimsional array of any internal type.

Returns:
- the array's internal type.
*/
get_type :: proc(mdarray: MdArray($T, $Nd)) -> typeid {
	return T
}


/*
Checks if the internal buffer of the array is empty.

Inputs:
- mdarray: a multidimsional array of any internal type.

Returns:
- a boolean signifying that the array is empty.
*/

is_none :: proc(mdarray: MdArray($T, $Nd)) -> bool {
	return slice.is_empty(mdarray.buffer)
}


/*
Properly creates an instance of a multi-dimensional array based on the provided
type and shape. The values are not assigned and will therefore follow the zero-value
of the provided type.


Inputs:
- T: the type of the array to be created.
- shape: an Odin array that holds the shape of the created multi-dimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: The created multi-dimensional array.
- ok: an optional boolean for error handling.
*/
make_mdarray :: proc(
	$T: typeid,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	size := validate_shape_and_get_size(shape, location=location) or_return
	buffer, err := make([]T, size)

	if err != .None {
		logging.error(.AllocationError, location = location)
		return {}, false
	}

	mdarray = MdArray(T, Nd) {
		buffer  = buffer,
		shape   = shape,
		strides = compute_strides(shape),
		is_view = false,
	}

	return mdarray, true
}

/*
Frees the internal memory of the array. The provided array will become empty. You sould
no longer use a freed array.

NOTE: This procedure will skip a view array. A warning will be logged as a reminder.

Inputs:
- mdarray: a multidimsional array of any internal type.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- allocation error value. .None implies that no error has occured.
*/
free_mdarray :: proc(
	mdarray: MdArray($T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> runtime.Allocator_Error {

	if mdarray.is_view {
		logging.warning(
			.OperationSkipped,
			"Attempted to free a view of an array. Operation skipped.",
			location=location
		)
		return .None		
	}

	return delete(mdarray.buffer, allocator, location)
}


/*
Create a multi-dimensional array from an Odin slice. The size of the array and the
shape provided have to be compatible. In other words the total size of the array should
be equal to the multiple of all dimensions in the shape.

NOTE: this procedure will create a copy of the provided slice. It does not create a view.

Inputs:
- sl: the slice from which the array will be created.
- shape: the shape of the multi-dimensional array. Must be compatible with the size of the slice.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array with same type as the provided slice.
- ok: an optional boolean for error handling.
*/
from_slice :: proc(
	sl: []$T,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	size := validate_shape_and_get_size(shape, location=location) or_return

	if len(sl) != size {
		logging.error(
			.ArguementError,
			"Provided shape is inconsistent with the length of the slice",
			location = location,
		)
		return {}, ok
	}

	mdarray = make_mdarray(T, shape, allocator, location) or_return
	copy(mdarray.buffer, sl)
	return mdarray, true
}


/*
Create a multi-dimensional array based on the provided shape, whose elements are filled
with the provided value.

NOTE: the internal type of the array corresponds to the type of the provided value.

Inputs:
- value: the value based on which the array will be filled 
- shape: the shape of the multi-dimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array filled with the provided value.
- ok: an optional boolean for error handling.
*/
fills :: proc(
	value: $T,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_positive_shape(shape, location=location) or_return
	
	mdarray = make_mdarray(T, shape, allocator, location) or_return

	for i in 0 ..< len(mdarray.buffer) {
		mdarray.buffer[i] = value
	}

	return mdarray, true
}


/*
Create a multi-dimensional array with an identical shape to another array, whose elements
are filled with the provided value.

Inputs:
- value: the value based on which the array will be filled 
- source: the array based on which the shape of the created array is decided. Its internal
type doesn't matter. The created array will follow the type of the provided value instead.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array filled with the provided value.
- ok: an optional boolean for error handling.
*/
fills_like :: proc(
	value: $T,
	source: MdArray($S, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {
	mdarray, ok = fills(value, source.shape, allocator, location = location)
	return 
}


/*
Create a multi-dimensional array based on the provided shape, whose elements are all zero.

Inputs:
- T: the type of created array. 
- shape: the shape of the multi-dimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array filled with zeros.
- ok: an optional boolean for error handling.
*/
zeros :: proc(
	$T: typeid,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	when intrinsics.type_is_complex(T){
		return fills(complex(0, 0), shape, allocator, location = location)
	} else {
		return fills(cast(T)0, shape, allocator, location = location)
	}
}


/*
Create a multi-dimensional array with an identical shape to another array, whose elements
are all with zeros.

Inputs:
- T: the type of created array. 
- source: the array based on which the shape of the created array is decided. Its internal
type doesn't matter. The created array will follow the type above.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array filled with zeros.
- ok: an optional boolean for error handling.
*/
zeros_like :: proc(
	$T: typeid,
	source: MdArray($S, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	when intrinsics.type_is_complex(T){
		return fills(complex(0, 0), source.shape, allocator, location = location)
	} else {
		return fills(cast(T)0, source.shape, allocator, location = location)
	}
}


/*
Create a multi-dimensional array based on the provided shape, whose elements are all ones.

Inputs:
- T: the type of created array. 
- shape: the shape of the multi-dimensional array.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array filled with ones.
- ok: an optional boolean for error handling.
*/
ones :: proc(
	$T: typeid,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	when intrinsics.type_is_complex(T){
		return fills(complex(1, 0), shape, allocator, location = location)
	} else {
		return fills(cast(T)1, shape, allocator, location = location)
	}
}


/*
Create a multi-dimensional array with an identical shape to another array, whose elements
are all with ones.

Inputs:
- T: the type of created array. 
- source: the array based on which the shape of the created array is decided. Its internal
type doesn't matter. The created array will follow the type above.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- mdarray: a multidimsional array filled with ones.
- ok: an optional boolean for error handling.
*/
ones_like :: proc(
	$T: typeid,
	source: MdArray($S, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	mdarray, ok = fills(cast(T)1, source.shape, allocator, location = location)
	return 
}



@(private="file")
inner_range :: proc(
	buffer: []$T,
	begin, step: T,
	size:int,
) {
	curr_val := begin
	for i in 0..< size {
		buffer[i] = curr_val
		curr_val+=step
	}
}


from_range :: proc(
	$T: typeid,
	end: int,
	begin:= 0,
	step:= 1,	
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, 1),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {

	md_size := (end - begin)/step
	md_size += ((end - begin)%step) == 0? 0 : 1
	
	mdarray = make_mdarray(T, [1]int{md_size}, allocator, location) or_return

	inner_range(mdarray.buffer, T(begin), T(step), size(mdarray))
	return mdarray, true
}


reshaped_range :: proc(
	$T: typeid,
	shape: [$Nd]int,
	begin:= 0,
	step:= 1,	
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {

	validate_positive_shape(shape, location=location) or_return

	mdarray = make_mdarray(T, shape, allocator, location) or_return

	inner_range(mdarray.buffer, T(begin), T(step), size(mdarray))
	return mdarray, true
}


@(private="file")
get_step :: #force_inline proc "contextless"(begin, end:$T, n: int, endpoint:bool) -> T {
	denom := endpoint? T(n)-1 : T(n)
	return (end - begin)/denom
}


linspace :: proc(
	$T: typeid,
	begin: T,
	end: T,
	n: int,
	endpoint:=true, 
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, 1),
	ok: bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) #optional_ok {
	step := get_step(begin, end, n, endpoint)
	
	mdarray = make_mdarray(T, [1]int{n}, allocator, location) or_return

	inner_range(mdarray.buffer, T(begin), T(step), size(mdarray))
	return mdarray, true
}


logspace :: proc(
	$T: typeid,
	begin: T,
	end: T,
	n: int,
	base:f64=10.0,
	endpoint:=true, 
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, 1),
	ok: bool,
) where intrinsics.type_is_float(T) #optional_ok {
	step := get_step(begin, end, n, endpoint)
	
	mdarray = make_mdarray(T, [1]int{n}, allocator, location) or_return

	curr_val := begin
	t_base := cast(T)base
	for i in 0..< size(mdarray) {
		mdarray.buffer[i] = math.pow(t_base, curr_val)
		curr_val+=step
	}
	return mdarray, true
}


eye :: proc(
	$T: typeid,
	n_rows: u64,
	n_cols : u64 = 0,
	diag_idx :int = 0,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, 2),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	n_cols := cast(int)n_cols
	n_rows := cast(int)n_rows
	abs_di := abs(diag_idx)

	if abs_di > n_cols {
		logging.error(.ArguementError, "diag_idx cannot be bigger than the number of cols.", location = location)
		return {}, ok
	}

	if n_cols == 0 {
		n_cols = n_rows
	}

	mdarray = zeros(T, [2]int{n_rows, n_cols}, allocator, location = location) or_return

	i := diag_idx if diag_idx >= 0 else n_cols * abs_di

	for _ in abs_di ..< min(n_rows, n_cols) {
		mdarray.buffer[i] = cast(T)1
		i += n_cols
		i += 1
	}

	return mdarray, true
}


identity :: proc(
	$T: typeid,
	n: u64,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, 2),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	return eye(T, n, n, diag_idx=0, allocator=allocator, location=location)
}


copy_array :: proc(
	source: MdArray($T, $Nd),
	allocator := context.allocator,	
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(source, location) or_return

	mdarray= make_mdarray(T, source.shape, allocator, location = location) or_return

	if !source.is_view {
		copy(mdarray.buffer, source.buffer)
		return mdarray, true
	}

	idx := 0
	for i in 0..<size(mdarray){
		idx = move_through_strides(i, mdarray.strides, source.strides)
		idx += source.offset
		mdarray.buffer[i] = source.buffer[idx]
	}
	
	return mdarray, true
}


cast_array :: proc(
	source: MdArray($T, $Nd),
	$to_type: typeid,
	allocator := context.allocator,
	location:=#caller_location,
) -> (
	mdarray: MdArray(to_type, Nd),
	ok: bool,
) where (intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T)) &&
 		(intrinsics.type_is_numeric(to_type) || intrinsics.type_is_boolean(to_type)) #optional_ok {

	validate_initialized(source, location) or_return

	mdarray = make_mdarray(to_type, source.shape, allocator, location) or_return
	
	if !source.is_view {
		for i in 0 ..< size(mdarray) {
			mdarray.buffer[i] = cast(to_type)source.buffer[i]
		}
		return mdarray, true
	}

	idx := 0
	for i in 0..<size(mdarray){
		idx = move_through_strides(i, mdarray.strides, source.strides)
		idx += source.offset
		mdarray.buffer[i] = cast(to_type)source.buffer[idx] 
	}

	return mdarray, true
}


reshape_view :: proc(
	mdarray: MdArray($T, $Nd),
	shape: [$Md]int,
	location := #caller_location,
) -> (
	result:MdArray(T, Md),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return

	if mdarray.is_view {
		logging.error(
			.NotImplemented,
			"reshaping a view array is not supported yet.",
			location
		)
	}

	shape := shape
	neg_axis := -1
	new_size := 1

	// check if it has -1
	for dim, i in shape {
		if dim != -1 && dim > 0 {
			new_size *= dim
		} else if dim == -1 && neg_axis == -1 {
			neg_axis = i
		} else {
			logging.error(
				.ArguementError,
				"Multiple -1 indices have been provided.",
				location = location,
			)
			return
		}
	}

	// infer the shape from -1
	if neg_axis != -1 {
		if size(mdarray) % new_size != 0 {
			logging.error(
				.ArguementError,
				"Inferring the size of the array is not possible with the provided shape.",
				location = location,
			)
			return
		}

		shape[neg_axis] = size(mdarray) / new_size
		new_size *= shape[neg_axis]
	}

	check_size := 1
	for d in shape do check_size *= d

	if size(mdarray) != check_size {
		logging.error(
			.ArguementError,
			"The shape provided is inconsistant with the size of the array.",
			location = location,
		)
		return
	}

	result = MdArray(T, Md) {
			buffer = mdarray.buffer,
			shape = shape,
			strides = compute_strides(shape),
			is_view=true,
			offset = mdarray.offset,
		}

	result.shape_strides = result.strides
	return result, true
}


reshape_copy :: proc(
	mdarray: MdArray($T, $Nd),
	shape: [$Md]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Md),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return

	temp_view := reshape_view(mdarray, shape, location) or_return
	result = make_mdarray(T, temp_view.shape, allocator, location) or_return

	if mdarray.is_view{
		for i in 0..<size(result){
			result.buffer[i] = get_linear(mdarray, i)
		}
	} else {			
		copy(result.buffer, mdarray.buffer)
	}

	return result, true
}


expand_dim_view :: proc(
	$Nd: int,
	mdarray: MdArray($T, Nd),
	axis:int,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd+1),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return
	if axis > Nd {
		logging.error(
			.ArguementError,
			"Provided axis cannot be larger than the dimensions of the array",
			location
		)
		return
	}

	new_shape := [Nd+1]int{}
	new_shape[axis] = 1
	offset:=0
	for d in 0..<Nd{
		if d == axis do offset = 1
		new_shape[d+offset] = mdarray.shape[d]
	}

	
	result = MdArray(T, Nd+1) {
			buffer = mdarray.buffer,
			shape = new_shape,
			strides = compute_strides(new_shape),
			is_view=true,
			offset = mdarray.offset,
		}

	result.shape_strides = result.strides
	return result, true
}


expand_dim_copy :: proc(
	$Nd: int,
	mdarray: MdArray($T, Nd),
	axis:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd+1),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return

	temp_view := expand_dim_view(Nd, mdarray, axis, location) or_return

	result = copy_array(temp_view, allocator, location) or_return
	
	return result, true
}


flatten_view :: proc(
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	result:MdArray(T, 1),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return

	if mdarray.is_view{
		logging.error(
			.NotImplemented,
			"flattening a view is not supported yet.",
			location)
		return 
	}

	result = MdArray(T, 1) {
			buffer = mdarray.buffer,
			shape = [1]int{size(mdarray)},
			strides = [1]int{1},
			is_view = true,
			offset = mdarray.offset,
		}

	result.shape_strides = result.strides
	return result, true
}


flatten_copy :: proc(
	mdarray: MdArray($T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, 1),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return

	result = make_mdarray(T, [1]int{size(mdarray)}, allocator) or_return
	if mdarray.is_view{
		for i in 0..<size(result){
			result.buffer[i] = get_linear(mdarray, i)
		}
	} else {			
		copy(result.buffer, mdarray.buffer)
	}
	return result, true
}


broadcast_to :: proc(
	mdarray: MdArray($T, $Nd),
	shape: [$Md]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, Md),
	ok:	bool,
) where intrinsics.type_is_numeric(T)|| intrinsics.type_is_boolean(T) , Md >= Nd #optional_ok {

	validate_initialized(mdarray, location) or_return

	curr_shape :[Md]int
	when Nd != Md{
		for &dim in curr_shape {dim = 1}
		offset:= Md-Nd
		for i in offset..<Md{
			curr_shape[i] = mdarray.shape[i-offset]
		}
	}else{
		curr_shape = mdarray.shape
	}

	sim_cnt:=0
	for i in 0..<Md {
		di := curr_shape[i]
		dj := shape[i]
		if di != 1 && di != dj {
			logging.error(.ArguementError, "Cannot broadcast to the provided shape.", location=location)
			return
		}
		if di == dj{
			sim_cnt += 1
		}
	}

	when Nd == Md {
 		if sim_cnt == Md {		
			logging.warning(
				.OperationSkipped,
				"provided shape is identical. The array has been returned unchanged.",
				location = location
			)
			return mdarray, true
		}
	}

	result = make_mdarray(T, shape, allocator, location) or_return

	reshaped_mdarr := reshape_view(mdarray, curr_shape, location) or_return
	pos : [Md]int
	for i in 0..<size(result){
		pos = from_buffer_index(result, i, location) or_return
		for d in 0..<Md {
			if curr_shape[d] == 1 do pos[d] = 0
		}
		result.buffer[i] = get(reshaped_mdarr, pos, location)
	}

	return result, true
}


broadcast_map :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	f: proc(T, T, ..$S) -> $R,
	args: ..S, 
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(R, Nd),
	ok:	bool,
) where intrinsics.type_is_numeric(T)|| intrinsics.type_is_boolean(T),
		intrinsics.type_is_numeric(S)|| intrinsics.type_is_boolean(S) #optional_ok
{

	result_shape: [Nd]int = broadcast_shape(a, b, location) or_return

	result = make_mdarray(S, result_shape, allocator, location)

	a_pos : [Nd]int
	b_pos : [Nd]int
	a_val : T
	b_val : T
	for i in 0..<size(result){
		a_pos = from_buffer_index(result, i, location) or_return
		b_pos = a_pos
		for d in 0..<Nd {
			if a.shape[d] == 1 do a_pos[d] = 0
			if b.shape[d] == 1 do b_pos[d] = 0
		}
		a_val = get(a, a_pos, location) or_return
		b_val = get(b, b_pos, location) or_return
		result.buffer[i] = f(a_val, b_val, ..args)
	}	

	return result, true
}


broadcast_shape :: proc(
	a: MdArray($T, $Nd),
	b: MdArray($S, Nd),
	location := #caller_location,
) -> (
	result: [Nd]int,
	ok:	bool,
) where 
	intrinsics.type_is_numeric(T)|| intrinsics.type_is_boolean(T),
	intrinsics.type_is_numeric(S)|| intrinsics.type_is_boolean(S) #optional_ok {

	validate_initialized(a, location) or_return
	validate_initialized(b, location) or_return

	for d in 0..<Nd {
		da := a.shape[d]
		db := b.shape[d]

		if da==db || da == 1 {
			result[d] = db
			continue
		}

		if db == 1{
			result[d] = da
			continue
		}

		logging.error(.ArguementError, "Cannot broadcast to the provided shape.", location=location)
		return
	}

	return result, true
}


stack :: proc(
	$Nd: int,
	mdarrays: []MdArray($T, Nd),
	axis := 0,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd + 1),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	first_shape := mdarrays[0].shape

	if axis > Nd {
		logging.error(.ArguementError,"The axis should not be bigger the number of dimensions.", location = location)
		return
	}
	
	for i in 0..<len(mdarrays){
		validate_initialized(mdarrays[i], location) or_return

		if i== 0 {continue}

		if first_shape != mdarrays[i].shape{
			logging.error(.ArguementError, "Cannot stack arrays of different shapes.", location=location)
			return
		}	
	}
	
	result_shape := [Nd + 1]int{}
	reshape_shape := [Nd+ 1]int{}

	result_shape[axis] = len(mdarrays)
	reshape_shape[axis] = 1

	offset:=0
	for i in 0..<Nd{
		if i == axis {offset=1}
		result_shape[i+offset] = first_shape[i]
		reshape_shape[i+offset] = first_shape[i]
	}

	result = make_mdarray(T, result_shape, allocator, location)
	adjusted_strides := compute_strides(reshape_shape)

	repeat := len(mdarrays)	
	for i in 0..<size(mdarrays[0]){
		buf_idx := move_through_strides(i, adjusted_strides, result.strides)
		for r in 0..<repeat {
			result.buffer[buf_idx + r * result.strides[axis]] = get_linear(mdarrays[r], i, location)
		}
	}

	return result, true
}


vstack :: proc(
	mdarrays: []MdArray($T, 1),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, 2),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	for i in 0..<len(mdarrays){
		validate_initialized(mdarrays[i], location) or_return
	}
	
	first_shape := mdarrays[0].shape

	for i in 1..<len(mdarrays){
		if first_shape != mdarrays[i].shape{
			logging.error(.ArguementError, "Cannot stack arrays of different shapes.", location=location)
			return
		}	
	}

	result_shape := [2]int{}
	result_shape[0] = len(mdarrays)
	result_shape[1] = first_shape[0]

	reshape_shape := [2]int{}
	reshape_shape[0] = 1
	reshape_shape[1] = first_shape[0]

	result = make_mdarray(T, result_shape, allocator, location)	
	adjusted_strides := compute_strides(reshape_shape)

	repeat := len(mdarrays)
	for i in 0..<size(mdarrays[0]){
		buf_idx := move_through_strides(i, adjusted_strides, result.strides)

		for r in 0..<repeat {
			result.buffer[buf_idx + r * result.strides[0]] = get_linear(mdarrays[r], i, location)
		}
	}

	return result, true
}


hstack :: proc(
	mdarrays: []MdArray($T, 1),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, 1),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {
	for i in 0..<len(mdarrays){
		validate_initialized(mdarrays[i], location) or_return
	}
	
	result_shape := [1]int{}
	first_shape := mdarrays[0].shape
	result_shape[0] = first_shape[0]

	for i in 1..<len(mdarrays){
		if first_shape != mdarrays[i].shape{
			logging.error(.ArguementError, "Cannot stack arrays of different shapes.", location=location)
			return
		}
		result_shape[0] += mdarrays[i].shape[0]
	}

	result = make_mdarray(T, result_shape, allocator, location)	

	cnt:=0
	r:=0
	for i in 0..<size(result){
		if cnt == size(mdarrays[r]){
			cnt = 0
			r+=1
		}
		result.buffer[i] = get_linear(mdarrays[r], cnt, location)	
		cnt += 1
	}

	return result, true
}


concat :: proc(
	mdarrays: []MdArray($T, $Nd),
	axis := 0,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {

	if axis > Nd {
		logging.error(
			.ArguementError,
			"The axis should not be bigger the number of dimensions.",
			location = location
		)
		return
	}

	first_shape := mdarrays[0].shape

	for i in 0..<len(mdarrays){

		validate_initialized(mdarrays[i], location) or_return

		if i== 0 {continue}

		for d in 0..<Nd{
			if d == axis {continue}
			if first_shape[d] != mdarrays[i].shape[d]{
				logging.error(.ArguementError, "Shapes of provided arrays must be identical (with the exception of the axis dimension).", location=location)
				return
			}
		}
	}

	
	result_shape := first_shape
	result_shape[axis] = 0
	for &mdarray in mdarrays{
		result_shape[axis] += mdarray.shape[axis]
	}

	result = make_mdarray(T, result_shape, allocator, location)

	axis_offset := 0
	md_idx : [Nd]int
	buf_idx := 0
	for m in 0..<len(mdarrays){
		for i in 0..<size(mdarrays[m]){
			md_idx = from_buffer_index(mdarrays[m], i, location) or_return
			md_idx[axis] += axis_offset
			buf_idx = to_buffer_index(result, md_idx, location) or_return
			
			result.buffer[buf_idx] = get_linear(mdarrays[m], i, location)
		}
		axis_offset += mdarrays[m].shape[axis]
	}

	return result, true
}


where_cond :: proc(
	mdarray: MdArray($T, $Nd),
	where_array: MdArray(bool, Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(T, 1),
	ok:	bool,
) where intrinsics.type_is_numeric(T)|| intrinsics.type_is_boolean(T) #optional_ok {

	validate_initialized(mdarray, location) or_return
	validate_initialized(where_array, location) or_return

	validate_shape_match(mdarray, where_array, location) or_return

	cnt:int
	for i in 0..<size(where_array){
		if get_linear(where_array, i) { cnt += 1}
	}

	result = make_mdarray(T, [1]int{cnt}, allocator, location) or_return

	cnt = 0
	for i in 0..<size(where_array){
		if get_linear(where_array, i) {
			result.buffer[cnt] = get_linear(mdarray, i)
			cnt +=1
		}
	}

	return result, true
}


//aliases
is_empty :: is_none
vflatten :: flatten_view
cflatten :: flatten_copy
vreshape :: reshape_view
creshape :: reshape_copy
