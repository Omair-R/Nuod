package fft

import fftod "./fftod"
import md "../mdarray"


fft_default :: proc(
	mdarray: md.MdArray($T, $Nd),
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) #optional_ok{
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
) #optional_ok{
	w_arr, err := make([]T, mdarray.shape[axis], allocator, location)
	if err != .None do return

	defer delete(w_arr)

	n_passes := 1
	for d in 0..<Nd{
		if d==axis do continue
		n_passes *= mdarray.shape[d]
	}

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
	fft_default,
	fft_with_axes
}


ifft_default :: proc(
	mdarray: md.MdArray($T, $Nd),
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, Nd),
	ok: bool,
) #optional_ok{
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
) #optional_ok{
	return fft_with_axes(mdarray, axis, true, allocator, location)
}

ifft :: proc{
	ifft_default,
	ifft_with_axes
}
