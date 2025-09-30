package mdarray

import errors "../errors"
import logging "../logging"
import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:slice"


MdArray :: struct($T: typeid, $Nd: int) {
	buffer:  []T,
	shape:   [Nd]int,
	strides: [Nd]int,

	// View specific variables
	is_view: bool,
	offset: int,
	shape_strides: [Nd]int,
}


ndim :: proc(mdarray: MdArray($T, $Nd)) -> int {
	return Nd
}

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

ravel :: proc(mdarray: MdArray($T, $Nd)) -> []T {
	return mdarray.buffer[:]
}

get_type :: proc(mdarray: MdArray($T, $Nd)) -> typeid {
	return type_of(T)
}

is_none :: proc(mdarray: MdArray($T, $Nd)) -> bool {
	return slice.is_empty(mdarray.buffer)
}


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


fills_like :: proc(
	value: $T,
	source: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) #optional_ok {
	mdarray, ok = fills(value, source.shape, allocator, location = location)
	return 
}


zeros :: proc(
	$T: typeid,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	mdarray, ok = fills(cast(T)0, shape, allocator, location = location)
	return 
}


zeros_like :: proc(
	$T: typeid,
	source: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	mdarray, ok = fills(cast(T)0, source.shape, allocator, location = location)
	return 
}


ones :: proc(
	$T: typeid,
	shape: [$Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	mdarray, ok = fills(cast(T)1, shape, allocator, location = location)
	return 
}


ones_like :: proc(
	$T: typeid,
	source: MdArray(T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	mdarray: MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_numeric(T) #optional_ok {
	mdarray, ok = fills(cast(T)1, source.shape, allocator, location = location)
	return 
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

	i:=0
	curr_val := begin
	for curr_val < end {
		mdarray.buffer[i] = cast(T)curr_val

		curr_val+=step
		i+=1
	}

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

	curr_val := begin
	for i in 0..<size(mdarray) {
		mdarray.buffer[i] = cast(T)curr_val
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
			return {}, false
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
			return {}, false
		}

		shape[neg_axis] = size(mdarray) / new_size
		new_size *= shape[neg_axis]
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

	temp_view := expand_dim_view(mdarray, shape, axis, location) or_return

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
		logging.warning("Not implemented yet.", location = location)
		return 
	}

	result = MdArray(T, Md) {
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
	if Nd != Md{
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

	if sim_cnt == Md {		
		logging.warning(
			.OperationSkipped,
			"provided shape is identical. The array has been returned unchanged.",
			location = location
		)
		return mdarray, true
	}

	result = make_mdarray(T, shape, allocator, location) or_return

	adjusted_strides := compute_strides(curr_shape)
	
	for i in 0..<size(mdarray){
		buf_idx := move_through_strides(i, adjusted_strides, result.strides)

		result.buffer[buf_idx] = get_linear(mdarray, i, location)
		for d in 0..<Md {
			di := curr_shape[d]
			dj := shape[d]

			if di != 1 || di == dj { continue }

			repeat := dj

			for r in 1..<repeat {
				result.buffer[buf_idx + r * result.strides[d]] = get_linear(mdarray, i, location)		
			}
		}
	}

	return result, true
}


broadcast_map :: proc(
	a: MdArray($T, $Nd),
	b: MdArray(T, Nd),
	f: proc(T, T, ..$S) -> S,
	args: ..S, 
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result: MdArray(S, Nd),
	ok:	bool,
) where intrinsics.type_is_numeric(T)|| intrinsics.type_is_boolean(T),
		intrinsics.type_is_numeric(S)|| intrinsics.type_is_boolean(S) #optional_ok
{

	result_shape: [Nd]int = broadcast_shape(a, b, location) or_return

	result = make_mdarray(S, result_shape, allocator, location)

	
	for i in 0..<size(a){
		buf_idx := move_through_strides(i, a.strides, result.strides)

		result.buffer[buf_idx] = get_linear(a, i, location)
		for d in 0..<Nd {
			di := a.shape[d]
			dj := result_shape[d]

			if di != 1 || di == dj { continue }

			for r in 1..<dj {
				result.buffer[buf_idx + r * result.strides[d]] = get_linear(a, i, location)		
			}
		}
	}
	
	
	for i in 0..<size(b){
		buf_idx := move_through_strides(i, b.strides, result.strides)

		val := get_linear(b, i)
		result.buffer[buf_idx] = f(result.buffer[buf_idx], val, ..args)
		for d in 0..<Nd {
			di := b.shape[d]
			dj := result_shape[d]

			if di != 1 || di == dj { continue }

			for r in 1..<dj {
				val_r := result.buffer[buf_idx + r * result.strides[d]]		
				result.buffer[buf_idx + r * result.strides[d]] = f(val_r, val, ..args)		
			}
		}
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
