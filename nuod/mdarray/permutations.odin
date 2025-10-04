package mdarray

import "base:intrinsics"
import "base:runtime"
import "core:math"

import "../logging"


inplace_matrix_transpose ::proc(
	mdarray: ^MdArray($T, 2),
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok:bool) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T) {

	validate_initialized(mdarray^, location) or_return

	rows := mdarray.shape[0]
	cols := mdarray.shape[1]

	if rows == cols {
		for i in 0..<(rows-1) {
			for j in i+1..<rows{
				ind_a := i*cols+j
				ind_b := j*cols+i
				swap(&mdarray.buffer[ind_a], &mdarray.buffer[ind_b])
			}
		}
		mdarray.shape = {mdarray.shape[1], mdarray.shape[0]}
		return true
	}

	m := rows
	n := cols

	g := math.gcd(m, n)
	l := max(m, n)

	a := m/g
	b := n/g
	
	temp, err := make([]T, l)
	if err != .None {
		logging.error(.AllocationError, location = location)
		return
	}

	defer delete(temp)

	pos := 0 

	if g > 1 {
		rot_i := 0
		for j in 0..<n{
			for i in 0..<m{
				rot_i = (i + j/b)%m
				pos = rot_i*cols+j
				temp[i] = mdarray.buffer[pos]
			}
			for i in 0..<m {
				pos = i*cols+j
				mdarray.buffer[pos] = temp[i]
			}
		}
	}

	sct_j := 0
	for i in 0..<m {
		for j in 0..<n {
			sct_j = ((i+j/b)% m + j*m)%n
			pos = i*cols+j
			temp[sct_j] = mdarray.buffer[pos]
		}
		for j in 0..<n{
			pos=i*cols+j
			mdarray.buffer[pos] = temp[j]
		}
	}
	
	gth_i := 0
	for j in 0..<n {
		for i in 0..<m{
			gth_i = (j+i*n-i/a)%m
			pos = gth_i*cols+j
			temp[i] = mdarray.buffer[pos]
		}
		for i in 0..<m{
			pos = i*cols+j
			mdarray.buffer[pos] = temp[i]
		}
	}
	
	mdarray.shape = {mdarray.shape[1], mdarray.shape[0]}
	return true
}


permute_dims_view :: proc(
	mdarray: MdArray($T, $Nd),
	indices:[Nd]int,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T), Nd>=2  #optional_ok {

	validate_initialized(mdarray, location) or_return

	for i in 0..<Nd {
		if indices[i] < 0 || indices[i] >= Nd {
			logging.error(
				.ArguementError,
				"Indices should be unique and within the limits of the array's dims.",
				location
			)
		}
	}
	
	new_shape : [Nd]int
	new_strides : [Nd]int

	for i in 0..<ndim(mdarray){
		j:= indices[i]
		new_shape[i] = mdarray.shape[j]
		new_strides[i] = mdarray.strides[j]
	}

	result = MdArray(T, Nd) {
			buffer = mdarray.buffer,
			shape = new_shape,
			strides = new_strides,
			offset = mdarray.offset,
			is_view = true,
	}
	result.shape_strides = compute_strides(new_shape)
	return result, true
}


permute_dims_copy :: proc(
	mdarray: MdArray($T, $Nd),
	indices:[Nd]int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T), Nd>=2  #optional_ok {
	permuted_view := permute_dims_view(mdarray, indices, location) or_return
	result = copy_array(permuted_view, allocator, location) or_return
	return result, true
}


permute_default_view :: proc(
	mdarray: MdArray($T, $Nd),
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T), Nd>=2  #optional_ok {

	axes : [Nd]int
	j := 0
	for i:=Nd-1; i >= 0; i-=1{
		 axes[j]=i
		 j+=1
	}

	return permute_dims_view(mdarray, axes, location)	
}


permute_default_copy :: proc(
	mdarray: MdArray($T, $Nd),
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T), Nd>=2  #optional_ok {
	permuted_view := permute_default_view(mdarray, location) or_return
	result = copy_array(permuted_view, allocator, location) or_return
	return result, true
}



swap_axes_view :: proc(
	mdarray: MdArray($T, $Nd),
	axis1:int,
	axis2:int,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T), Nd>=2  #optional_ok {

	axes : [Nd]int
	for i in 0..<Nd{
		 axes[i]=i
	}

	axes[axis1] = axis2
	axes[axis2] = axis1
	
	result = permute_dims_view(mdarray, axes, location) or_return
	return result, true
}


swap_axes_copy :: proc(
	mdarray: MdArray($T, $Nd),
	axis1:int,
	axis2:int,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	result:MdArray(T, Nd),
	ok:bool,
) where intrinsics.type_is_numeric(T) || intrinsics.type_is_boolean(T), Nd>=2  #optional_ok {
	permuted_view := swap_axes_view(mdarray, axis1, axis2, location) or_return
	result = copy_array(permuted_view, allocator, location) or_return
	return result, true
}

vpermute_dims :: permute_dims_view
cpermute_dims :: permute_dims_copy
transpose_view :: proc{permute_dims_view, permute_default_view}
transpose_copy :: proc{permute_dims_copy, permute_default_copy}
vtranspose :: transpose_view
ctranspose :: transpose_copy
vswap_axes :: swap_axes_view
cswap_axes :: swap_axes_copy
