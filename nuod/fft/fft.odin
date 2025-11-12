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


/*
Compute the one-dimensional fast fourier transform for a one-dimensional signal.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a vector representing a complex signal.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
fft_vector :: proc(
	mdarray: md.MdArray($T, 1),
	inverse:=false,
	allocator:=context.allocator,
	location:= #caller_location,	
)-> (
	result: md.MdArray(T, 1),
	ok: bool,
) where intrinsics.type_is_complex(T) #optional_ok{

	md.validate_initialized(mdarray, location) or_return

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


/*
Compute the one-dimensional fast fourier transform for a multidimensional signal along
the last axis dimension.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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


/*
Compute the one-dimensional fast fourier transform for a multidimensional signal along
a selected axis dimension.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- axis: the axis along which the fourier transform is computed.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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
	
	md.validate_initialized(mdarray, location) or_return
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


/*
Compute the one-dimensional inverse fast fourier transform for a one-dimensional signal.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a vector representing a complex signal.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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


/*
Compute the one-dimensional inverse fast fourier transform for a complex multidimensional
signal along the last axis dimension.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the time domain.
- ok: an optional boolean for error handling.
*/
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


/*
Compute the one-dimensional inverse fast fourier transform for a complex multidimensional
signal along a specific axis dimension.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- axis: the axis along which the fourier transform is computed.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the time domain.
- ok: an optional boolean for error handling.
*/
ifft_with_axes :: proc(
	mdarray: md.MdArray($T, $Nd),
	axis:int,
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


/*
Compute the two-dimensional fast fourier transform for a multidimensional signal along
the last two axis dimensions.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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


/*
Compute the two-dimensional fast fourier transform for a multidimensional signal along two
specific axis dimensions.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- axes: an a array of two elements representing the axes along which the transform is performed.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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
	
	result = fft_with_axes(mdarray, axes[0], inverse, allocator, location) or_return
	fft_pass_inplace(result, axes[1], inverse, allocator, location)
	return result, true
}


fft2d :: proc{
	fft2d_default,
	fft2d_with_axes
}


/*
Compute the multi-dimensional fast fourier transform for a multidimensional signal along
all its axis dimensions.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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


/*
Compute the multi-dimensional fast fourier transform for a multidimensional signal along
a set of specific axis dimensions.

NOTE: Use of this procedure is discourged. Please use the procedure group instead.

Inputs:
- mdarray: a multidimensional array representing a complex signal.
- axes: the axes along which the transform is performed.
- inverse: a boolean to compute the inverse of the fft.
- allocator: the allocator used internally.
- location: a debugging variable used to trace the location of the calling procedure.

Returns:
- result: the signal in the frequency domain.
- ok: an optional boolean for error handling.
*/
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
