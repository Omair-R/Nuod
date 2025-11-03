package fftod

import "core:log"
import "base:intrinsics"
import "core:math"


@private
_inner_fft_stockham_radix2 :: proc (
	n:int,
	s: int,
	x : []$C,
	work:[]C,
	twiddles:[]C,
	flip: bool,
	mixed:=false,
) where intrinsics.type_is_complex(C) {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	m := n>>1

	if n == 2{
		z := flip ? work : x
		for q in 0..<s{
			a := x[q]
			b := x[q+s]
			z[q] = a + b
			z[q+s] = a - b
		}
	} else if n >= 4 {
		sm := s*m
		for q in 0..<s{
			a := x[q]
			b := x[q + sm]
			work[q] = a + b
			work[q + s] = a - b
		}
		wp := twiddles[s-1]
		for p in 1..<m {
			sp := s*p
			sp2 := sp<<1
			for q in 0..<s{
				a := x[q + sp]
				b := x[q + sp+ sm]
				work[q + sp2] = a + b
				work[q + sp2 + s] = (a - b) * wp
			}
			wp = twiddles[s+sp-1]
		}
		if mixed {
			_inner_fft_stockham_mixed(m, s<<1, work, x, twiddles, !flip)
		} else {
			_inner_fft_stockham_radix2(m, s<<1, work, x, twiddles, !flip)
		}
	}
}


out_fft_stockham_radix2_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.Radix2,
		x,
		plan,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix2_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.Radix2,
		x,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix2 :: proc{
	out_fft_stockham_radix2_planned,
	out_fft_stockham_radix2_unplanned
}


in_fft_stockham_radix2_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix2,
		x,
		plan,
		inverse,
		location
	)
}


in_fft_stockham_radix2_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix2,
		x,
		inverse,
		allocator,
		location
	)
}


in_fft_stockham_radix2 :: proc{
	in_fft_stockham_radix2_planned,
	in_fft_stockham_radix2_unplanned
}
