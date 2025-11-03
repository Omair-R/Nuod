package fftod

import "core:log"
import "base:intrinsics"
import "core:math"


@private
_inner_fft_stockham_radix3 :: proc (
	n:int,
	s: int,
	x : []$C,
	work:[]C,
	twiddles:[]C,
	flip: bool,
	mixed:= false,
) where intrinsics.type_is_complex(C) {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	n1 := n/3
	n2 := n1*2

	w3 := rect_complex(1, -F(math.TAU/3))
	w32 := w3 * w3

	if n==3 {
		z := flip ? work : x
		for q in 0..<s{
			a := x[q + s*(0)]
			b := x[q + s*(1)]
			c := x[q + s*(2)]
			z[q + s*( 0)] = a + b + c
			z[q + s*( 1)] = a + w3 * b + w32 * c
			z[q + s*( 2)] = a + w32 * b + w3 * c
		}
	} else if n > 3 {
		w1p := C(1+0i)
		w2p := w1p
		for p in 0..<n1 {
			p_3 := p*3
			sp := s*p
			sp_3 := sp*3
			for q in 0..<s{
				a := x[q + sp]
				b := x[q + sp+s*n1]
				c := x[q + sp+s*n2]
				work[q + sp_3] = a + b + c
				work[q + sp_3 + s] = (a + w3 * b + w32 * c) * w1p
				work[q + sp_3 + s*2] = (a + w32 * b + w3 * c) * w2p
			}
			w1p = twiddles[s+sp-1]
			w2p = twiddles[2*(s+sp)-1]
		}
		if mixed {
			_inner_fft_stockham_mixed(n1, s*3, work, x, twiddles, !flip)
		} else {
			_inner_fft_stockham_radix3(n1, s*3, work, x, twiddles, !flip)
		}
	}
}


out_fft_stockham_radix3_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.Radix3,
		x,
		plan,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix3_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{

	return out_fft_stockham(
		.Radix3,
		x,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix3 :: proc{
	out_fft_stockham_radix3_planned,
	out_fft_stockham_radix3_unplanned
}


in_fft_stockham_radix3_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix3,
		x,
		plan,
		inverse,
		location
	)
}


in_fft_stockham_radix3_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix3,
		x,
		inverse,
		allocator,
		location
	)
}


in_fft_stockham_radix3 :: proc{
	in_fft_stockham_radix3_planned,
	in_fft_stockham_radix3_unplanned
}
