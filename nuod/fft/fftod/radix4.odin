package fftod


import "core:log"
import "base:intrinsics"
import "core:math"


@private
_inner_fft_stockham_radix4 :: proc (
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
	
	n1 := n>>2
	n2 := n1<<1
	n3 := n1 + n2

	sn1 := s*n1
	sn2 := s*n2
	sn3 := s*n3

	if n == 4{
		z := flip ? work : x
		for q in 0..<s{
			a := x[q]
			b := x[q + s]
			c := x[q + s<<1]
			d := x[q + s*3]				
			apc := a + c
			amc := a - c
			bpd := b + d
			jbmd := 1i*(b-d)
			z[q] = apc + bpd
			z[q + s] = amc - jbmd
			z[q + s<<1] = apc - bpd
			z[q + s*3] = amc + jbmd
		}
	} else if n > 4 {
		w1p :C = 1+0i
		w2p :C = 1+0i
		w3p :C = 1+0i
		for p in 0..<n1 {
			p_4 := p<<2
			sp := s*p
			sp_4 := s*p_4
			if s== 1{
				a := x[p]
				b := x[p+n1]
				c := x[p+n2]
				d := x[p+n3]				
				apc := a + c
				amc := a - c
				bpd := b + d
				jbmd := 1i*(b - d)
				work[p_4] = apc + bpd
				work[p_4 + 1] = w1p * (amc - jbmd)
				work[p_4 + 2] = w2p * (apc - bpd)
				work[p_4 + 3] = w3p * (amc + jbmd)
			}
			else do for q in 0..<s{
				a := x[q + sp]
				b := x[q + sp+sn1]
				c := x[q + sp+sn2]
				d := x[q + sp+sn3]				
				apc := a + c
				amc := a - c
				bpd := b + d
				jbmd := 1i*(b - d)
				work[q + sp_4] = apc + bpd
				work[q + sp_4 + s] = w1p * (amc - jbmd)
				work[q + sp_4 + s<<1] = w2p * (apc - bpd)
				work[q + sp_4 + s*3] = w3p * (amc + jbmd)
			}
			w1p = twiddles[s+sp-1]
			w2p = twiddles[((s+sp)<<1)-1]
			w3p = twiddles[3*(s+sp)-1]
		}
		if mixed {
			_inner_fft_stockham_mixed(n1, s<<2, work, x, twiddles, !flip)
		} else {
			_inner_fft_stockham_radix4(n1, s<<2, work, x, twiddles, !flip)
		}
	}
}


out_fft_stockham_radix4_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	return out_fft_stockham(
		.Radix4,
		x,
		plan,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix4_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{

	return out_fft_stockham(
		.Radix4,
		x,
		inverse,
		allocator,
		location
	)
}


out_fft_stockham_radix4 :: proc{
	out_fft_stockham_radix4_planned,
	out_fft_stockham_radix4_unplanned
}


in_fft_stockham_radix4_planned :: proc(
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix4,
		x,
		plan,
		inverse,
		location
	)
}


in_fft_stockham_radix4_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	return in_fft_stockham(
		.Radix4,
		x,
		inverse,
		allocator,
		location
	)
}


in_fft_stockham_radix4 :: proc{
	in_fft_stockham_radix4_planned,
	in_fft_stockham_radix4_unplanned
}
