package fft

import "base:intrinsics"
import fftod "./fftod"
import md "../mdarray"
import "../logging"


@private
compute_passes_n :: proc(
	shape: [$Nd]int,
	axis: int,
) -> int {
	n_passes := 1
	for d in 0..<Nd{
		if d==axis do continue
		n_passes *= shape[d]
	}
	return n_passes	
}


@private
fft_pass_inplace :: proc(
	mdarray: md.MdArray($T, $Nd),
	axis: int,
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>=2 {
	w_arr, err := make([]T, mdarray.shape[axis], allocator, location)
	if err != .None do return

	defer delete(w_arr)

	n_passes := compute_passes_n(mdarray.shape, axis)

	if inverse {
		for i in 0..<n_passes{
			md.extract_linear_array(mdarray, w_arr, i, axis, location) or_return
			fftod.in_ifft(w_arr, allocator, location) or_return // TODO replace with plan based for speed
			md.placein_linear_array(mdarray, w_arr, i, axis, location) or_return
		}
	} else {
		for i in 0..<n_passes{
			md.extract_linear_array(mdarray, w_arr, i, axis, location) or_return
			fftod.in_fft(w_arr, allocator, location) or_return // TODO replace with plan based for speed
			md.placein_linear_array(mdarray, w_arr, i, axis, location) or_return
		}
	}
	return true
}


fft_vector :: proc(
	mdarray: md.MdArray($T, 1),
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, 1),
	ok: bool,
) where intrinsics.type_is_complex(T) #optional_ok{

	result = md.copy_array(mdarray, allocator, location) or_return

	if inverse {
		for i in 0..<n_passes{
			fftod.in_ifft(result.buffer, allocator, location) or_return // TODO replace with plan based for speed
		}
	} else {
		for i in 0..<n_passes{
			fftod.in_fft(result.buffer, allocator, location) or_return // TODO replace with plan based for speed
		}
	}

	return result, true
}


fft_default :: proc(
	mdarray: md.MdArray($T, $Nd),
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>1 #optional_ok{
	axis := Nd-1
	return fft_with_axes(mdarray, axis, inverse, allocator, location)
}


fft_with_axes :: proc(
	mdarray: md.MdArray($T, $Nd),
	axis:int,
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>1 #optional_ok{
	
	md.validate_axis(Nd, axis, location) or_return

	w_arr, err := make([]T, mdarray.shape[axis], allocator, location)
	if err != .None do return

	defer delete(w_arr)

	n_passes := compute_passes_n(mdarray.shape, axis)


	result = md.zeros_like(T, mdarray, allocator, location) or_return

	if inverse {
		for i in 0..<n_passes{
			md.extract_linear_array(mdarray, w_arr, i, axis, location) or_return
			fftod.in_ifft(w_arr, allocator, location) or_return // TODO replace with plan based for speed
			md.placein_linear_array(result, w_arr, i, axis, location) or_return
		}
	} else {
		for i in 0..<n_passes{
			md.extract_linear_array(mdarray, w_arr, i, axis, location) or_return
			fftod.in_fft(w_arr, allocator, location) or_return // TODO replace with plan based for speed
			md.placein_linear_array(result, w_arr, i, axis, location) or_return
		}
	}

	return result, true
}


fft :: proc{
	fft_vector,
	fft_default,
	fft_with_axes
}


ifft_vector :: proc(
	mdarray: md.MdArray($T, 1),
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, 1),
	ok: bool,
) where intrinsics.type_is_complex(T) #optional_ok{

	return fft_vector(mdarray, true, allocator, location)
}


ifft_default :: proc(
	mdarray: md.MdArray($T, $Nd),
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>1 #optional_ok{
	axis := Nd-1
	return fft_with_axes(mdarray, axis, true, allocator, location)
}


ifft_with_axes :: proc(
	mdarray: md.MdArray($T, $Nd),
	axis:int,
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>1 #optional_ok{
	return fft_with_axes(mdarray, axis, true, allocator, location)
}

ifft :: proc{
	ifft_vector,
	ifft_default,
	ifft_with_axes
}


fft2d_default :: proc(
	mdarray: md.MdArray($T, $Nd),
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>=2 #optional_ok{

	axes := [2]int{Nd-1, Nd-2}

	return fft2d_with_axes(mdarray, axes, inverse, allocator, location)
}


fft2d_with_axes :: proc(
	mdarray: md.MdArray($T, $Nd),
	axes : [2]int,
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>=2 #optional_ok{

	md.validate_axes(Nd, axes, location) or_return
	// first pass done horizontally. 
	result = fft_with_axes(mdarray, axes[0], inverse, allocator, location) or_return
	fft_pass_inplace(result, axes[1], inverse, allocator, location)
	return result, true
}


fft2d :: proc{
	fft2d_default,
	fft2d_with_axes
}


fftnd_default :: proc(
	mdarray: md.MdArray($T, $Nd),
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>=2 #optional_ok{

	axes : [Nd]int
	for d in 0..<Nd{
		axes[d] = Nd-(d+1)
	}

	return fftnd_with_axes(mdarray, axes, inverse, allocator, location)
}


// TODO: verfiy axes
fftnd_with_axes :: proc(
	mdarray: md.MdArray($T, $Nd),
	axes : [$Md]int,
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) where intrinsics.type_is_complex(T) && Nd>=2 && Md >=1 && Md <= Nd #optional_ok{

	md.validate_axes(Nd, axes, location) or_return
	// first pass done horizontally. 
	result = fft_with_axes(mdarray, axes[0], inverse, allocator, location) or_return

	for d in 1..<Md {
		fft_pass_inplace(result, axes[d], inverse, allocator, location)
	}
	return result, true
}


fftnd :: proc{
	fftnd_default,
	fftnd_with_axes
}
