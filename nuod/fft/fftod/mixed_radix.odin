package fftod


import "core:log"
import "base:intrinsics"
import "core:math"


@private
_inner_fft_stockham_mixed :: proc(
	n: int,
	s: int,
	x : []$C,
	work:[]C,
	twiddles:[]C,
	flip: bool,
) where intrinsics.type_is_complex(C) {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	if n % 4 == 0 {
		_inner_fft_stockham_radix4(
			n, s, x, work, twiddles, flip, mixed=true
		)
	} else if n % 2 == 0{
		_inner_fft_stockham_radix2(
			n, s, x, work, twiddles, flip, mixed=true
		)
	} else if n % 3 == 0{
		_inner_fft_stockham_radix3(
			n, s, x, work, twiddles, flip, mixed=true
		)
	} else if n % 5 == 0{
		_inner_fft_stockham_radix5(
			n, s, x, work, twiddles, flip, mixed=true
		)
	} else {
		// TODO: error 
	}
}


out_fft_stockham_mixed_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.MixedRadix,
		x,
		plan,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_mixed_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.MixedRadix,
		x,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_mixed :: proc{
	out_fft_stockham_mixed_planned,
	out_fft_stockham_mixed_unplanned
}


in_fft_stockham_mixed_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.MixedRadix,
		x,
		plan,
		inverse,
		location
	)
}


in_fft_stockham_mixed_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.MixedRadix,
		x,
		inverse,
		allocator,
		location
	)
}


in_fft_stockham_mixed :: proc{
	in_fft_stockham_mixed_planned,
	in_fft_stockham_mixed_unplanned
}
