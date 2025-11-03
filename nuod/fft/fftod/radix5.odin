package fftod


import "core:log"
import "base:intrinsics"
import "core:math"


@private
_inner_fft_stockham_radix5 :: proc (
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

	n1 := n/5
	n2 := n1*2
	n3 := n1*3
	n4 := n1*4

	w51 := rect_complex(1, -F(math.TAU/5))
	w52 := w51 * w51
	w53 := w52 * w51
	w54 := w52 * w52

	if n == 5 {
		z := flip ? work : x
		for q in 0..<s{
			a := x[q + s*(0)]
			b := x[q + s*(1)]
			c := x[q + s*(2)]
			d := x[q + s*(3)]
			e := x[q + s*(4)]
			z[q + s*(0)] = a + b + c + d + e
			z[q + s*(1)] = a + w51 * b + w52 * c + w53 * d + w54 * e
			z[q + s*(2)] = a + w52 * b + w54 * c + w51 * d + w53 * e
			z[q + s*(3)] = a + w53 * b + w51 * c + w54 * d + w52 * e
			z[q + s*(4)] = a + w54 * b + w53 * c + w52 * d + w51 * e
		}
	} else if n > 5 {
		w1p := C(1+0i)
		w2p := C(1+0i)
		w3p := C(1+0i)
		w4p := C(1+0i)
		for p in 0..<n1 {
			p_5 := p*5
			sp := s*p
			for q in 0..<s{
				a := x[q + s*(p+0)]
				b := x[q + s*(p+n1)]
				c := x[q + s*(p+n2)]
				d := x[q + s*(p+n3)]
				e := x[q + s*(p+n4)]
				work[q + s*(p_5 + 0)] = a + b + c + d + e
				work[q + s*(p_5 + 1)] = (a + w51 * b + w52 * c + w53 * d + w54 * e) * w1p
				work[q + s*(p_5 + 2)] = (a + w52 * b + w54 * c + w51 * d + w53 * e) * w2p
				work[q + s*(p_5 + 3)] = (a + w53 * b + w51 * c + w54 * d + w52 * e) * w3p
				work[q + s*(p_5 + 4)] = (a + w54 * b + w53 * c + w52 * d + w51 * e) * w4p
			}
			w1p = twiddles[s+sp-1]
			w2p = twiddles[2*(s+sp)-1]
			w3p = twiddles[3*(s+sp)-1]
			w4p = twiddles[4*(s+sp)-1]
		}
		if mixed {
			_inner_fft_stockham_mixed(n1, s*5, work, x, twiddles, !flip)
		} else {
			_inner_fft_stockham_radix5(n1, s*5, work, x, twiddles, !flip)
		}
	}
}


out_fft_stockham_radix5_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.Radix5,
		x,
		plan,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix5_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{

	return out_fft_stockham(
		.Radix5,
		x,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix5 :: proc{
	out_fft_stockham_radix5_planned,
	out_fft_stockham_radix5_unplanned
}


in_fft_stockham_radix5_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix5,
		x,
		plan,
		inverse,
		location
	)
}


in_fft_stockham_radix5_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix5,
		x,
		inverse,
		allocator,
		location
	)
}


in_fft_stockham_radix5 :: proc{
	in_fft_stockham_radix5_planned,
	in_fft_stockham_radix5_unplanned
}
