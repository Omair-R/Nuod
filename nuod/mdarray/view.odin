package mdarray

import "../logging"
import "base:intrinsics"

@private
to_buffer_index :: #force_inline proc(
	mdarray: MdArray($T, $Nd),
	indices:[Nd]int,
	location := #caller_location,
) -> (
	bf_ind:int,
	ok:bool, 
) where intrinsics.type_is_numeric(T) #optional_ok {

	when ODIN_DEBUG {			
		validate_initialized(mdarray, location) or_return
	}

	if mdarray.is_view {
		bf_ind += mdarray.offset
	} 

	for i in 0..<Nd {
		bf_ind += indices[i] * mdarray.strides[i]
	}
	return bf_ind, true
}


@private
from_buffer_index :: #force_inline proc(
	mdarray: MdArray($T, $Nd),
	bf_ind:int,
	location := #caller_location,
) -> (
	indices:[Nd]int,
	ok:bool, 
) where intrinsics.type_is_numeric(T) #optional_ok {
	
	when ODIN_DEBUG {			
		validate_initialized(mdarray, location) or_return
	}

	bf_ind := bf_ind

	if mdarray.is_view {
		bf_ind -= mdarray.offset
	} 

	for i in 0..<Nd {
		indices[i] = bf_ind / mdarray.strides[i]
		bf_ind %= mdarray.strides[i]
	}

	return indices, true		
}


@private
move_through_strides :: #force_inline proc(
	from_idx :int,
	from_strides: [$Nd]int,
	to_strides: [Nd]int,
) -> (
	to_idx:int,
) {
	from_idx := from_idx
	temp:int
	for i in 0..<Nd {
		temp = from_idx / from_strides[i]
		from_idx %= from_strides[i]
		to_idx += temp * to_strides[i]
	}

	return to_idx		
}


get_reduced_pos :: proc(
	shape: [$Nd]int,
	idx:int,
	axis:int
) -> (
	pos: [Nd]int,
) {
	
	reduced_shape: [Nd]int
	for d in 0..<Nd{
		if d==axis {
			reduced_shape[d] =1
			continue
		}
		reduced_shape[d] = shape[d]
	}

	strides:= compute_strides(reduced_shape)

	idx := idx
	for d in 0..<Nd{
		pos[d] = idx /strides[d] 
		idx %=strides[d]
	}

	return pos
}


extract_linear_array :: proc(
	mdarray: MdArray($T, $Nd),
	result: []T,
	idx:int, 
	axis:int,
	location:= #caller_location,
) -> (
	ok: bool
){

	when ODIN_DEBUG {			
		validate_initialized(mdarray, location) or_return
		if len(arr) != mdarray.shape[axis] {

			return
		}
	}

	n:= len(result) 
	last_i := Nd-1
	if last_i == axis && !mdarray.is_view {
		copy(result, mdarray.buffer[idx*n:(idx+1)*n])
	}

	pos := get_reduced_pos(mdarray.shape, idx, axis)

	for i in 0..<n{
		result[i] = get(mdarray, pos, location) or_return 
		pos[axis] += 1		
	}

	return true
}


placein_linear_array :: proc(
	mdarray: MdArray($T, $Nd),
	arr: []T,
	idx:int, 
	axis:int,
	location:= #caller_location,
) -> (
	ok: bool
){

	when ODIN_DEBUG {			
		validate_initialized(mdarray, location) or_return
		if len(arr) != mdarray.shape[axis] {

			return
		}
	}

	n:= len(arr) 
	last_i := Nd-1
	if last_i == axis && !mdarray.is_view {
		copy(mdarray.buffer[idx*n:(idx+1)*n], arr)
	}

	pos := get_reduced_pos(mdarray.shape, idx, axis)

	val : ^T
	for i in 0..<n{
		val = get_ref(mdarray, pos, location) or_return 
		val^ = arr[i]
		pos[axis] += 1		
	}

	return true
}


get_linear_ref :: #force_inline proc(
	mdarray : MdArray($T, $Nd),
	idx:int,
	location := #caller_location,
) -> (
	val :^T
) {
	bf_idx:= idx

	if mdarray.is_view{
		bf_idx = move_through_strides(idx, mdarray.shape_strides, mdarray.strides)
		bf_idx += mdarray.offset
	}	

	return &mdarray.buffer[bf_idx]
}


get_linear :: #force_inline proc(
	mdarray : MdArray($T, $Nd),
	idx:int,
	location := #caller_location,
) -> (
	val :T
) {
	bf_idx:= idx

	if mdarray.is_view{
		bf_idx = move_through_strides(idx, mdarray.shape_strides, mdarray.strides)
		bf_idx += mdarray.offset
	}	

	return mdarray.buffer[bf_idx]
}


get :: proc(	
	mdarray: MdArray($T, $Nd),
	pos: [Nd]int, 
	location := #caller_location,
) -> (
	val: T,
	ok: bool,
) #optional_ok {	

	validate_initialized(mdarray, location) or_return

	validate_pos_within_shape(pos, mdarray.shape) or_return 

	idx := to_buffer_index(mdarray, pos, location=location)
	val = mdarray.buffer[idx]

	return val, true 
}


get_ref :: proc(	
	mdarray: MdArray($T, $Nd),
	pos: [Nd]int, 
	location := #caller_location,
) -> (
	val: ^T,
	ok: bool,
) #optional_ok {	

	validate_initialized(mdarray, location) or_return

	validate_pos_within_shape(pos, mdarray.shape) or_return 

	idx := to_buffer_index(mdarray, pos, location=location)
	val = &mdarray.buffer[idx]

	return val, true 
}


narrow :: proc(
	mdarray: MdArray($T, $Nd),
	axis:int,
	begin:=0,
	end:=-1,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) #optional_ok {

	validate_initialized(mdarray, location) or_return

	if axis >= ndim(mdarray) {
		logging.error(.ArguementError, "Provided axis cannot be bigger than the number of dimensions.", location=location)
		return {}, ok
	}

	end := end
	if end == -1 {
		end = mdarray.shape[axis]
	}

	if begin >= end || end > mdarray.shape[axis]{
		logging.error(.ArguementError, "The narrow range provided is incorrect.", location=location)
		return {}, ok
	}

	shape := mdarray.shape
	shape[axis] = end - begin
	offset_plus := begin * mdarray.strides[axis]

	result = MdArray(T, Nd){
		buffer=mdarray.buffer,
		shape=shape,
		strides=mdarray.strides,
		offset=mdarray.offset+offset_plus,
		is_view=true,
	}

	result.shape_strides = compute_strides(shape)
	return result, true
}


slice_view :: proc(
	$Nd: int,
	mdarray: MdArray($T, Nd),
	index:int,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd-1),
	ok:bool,
) #optional_ok {

	validate_initialized(mdarray, location) or_return

	if index > mdarray.shape[0] {
		logging.error(.ArguementError, "Provided axis cannot be bigger than the number of dimensions.", location=location)
		return {}, ok
	}

	new_shape : [Nd-1]int
	new_strides : [Nd-1]int

	for d in 1..<ndim(mdarray){
		new_shape[d-1] = mdarray.shape[d]
		new_strides[d-1] = mdarray.strides[d]
	}

	offset_plus := index * mdarray.strides[0]

	result = MdArray(T, Nd-1){
		buffer=mdarray.buffer,
		shape=new_shape,
		strides=new_strides,
		offset=mdarray.offset+offset_plus,
		is_view=true,
	}

	result.shape_strides = compute_strides(new_shape)

	return result, true
}
