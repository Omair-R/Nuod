package fftod

import "core:math"
import "base:intrinsics"
import "base:runtime"


StockhamAlgorithm:: enum {
	Radix2,
	Radix3,
	Radix4,
	Radix5,
	MixedRadix,
}


Stockham_Plan :: struct($C:typeid)
where intrinsics.type_is_complex(C)
{
	n:int, 
	work: []C,
	twiddles: []C,
}


make_stockham_plan :: proc(
	n: int,
	$C: typeid,
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	plan: Stockham_Plan(C),
	err: runtime.Allocator_Error,
) where intrinsics.type_is_complex(C) #optional_allocator_error {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	work := make([]C, n, allocator, location) or_return
	twiddles := make([]C, n, allocator, location) or_return

	theta0 :F = math.TAU/F(n)

	for i in 0..<(n){
		twiddles[i] = rect_complex(1.0, -F(i+1)*theta0)
	}

	return Stockham_Plan(C){
		n,
		work,
		twiddles,
	}, .None
}


delete_stockham_plan :: proc(
	plan: Stockham_Plan($C),
	allocator:=context.allocator,
	location:= #caller_location,
) -> runtime.Allocator_Error {
	err := delete(plan.work, allocator, location)
	if err != .None do return err
	return delete(plan.twiddles, allocator, location)
}


verfiy_power :: proc(algo : StockhamAlgorithm, n:int) -> bool {
	
	switch algo {
		case .Radix2:
			return is_power2(n)
		case .Radix3:
			return is_power3(n)
		case .Radix4:
			return is_power4(n)
		case .Radix5:
			return is_power5(n)
		case .MixedRadix:
			return is_factorizable(n)
	}

	return false
}

out_fft_stockham_planned :: proc(
	algo: StockhamAlgorithm,
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	n := plan.n
	if len(x) != plan.n do return

	if !verfiy_power(algo, n) do return
	
	x_f = make([]C, n, allocator, location)
	copy(x_f, x)

	if n == 1 do return x_f, true
	
	if inverse do conj_iter(x_f)

	switch algo {
		case .Radix2:
			_inner_fft_stockham_radix2(n, 1, x_f, plan.work, plan.twiddles, flip=false)
		case .Radix3:
			_inner_fft_stockham_radix3(n, 1, x_f, plan.work, plan.twiddles, flip=false)
		case .Radix4:
			_inner_fft_stockham_radix4(n, 1, x_f, plan.work, plan.twiddles, flip=false)
		case .Radix5:
			_inner_fft_stockham_radix5(n, 1, x_f, plan.work, plan.twiddles, flip=false)
		case .MixedRadix:
			_inner_fft_stockham_mixed(n, 1, x_f, plan.work, plan.twiddles, flip=false)
	}

	if inverse do for &val in x_f do val /= complex(cast(F)n, 0)
	if inverse do conj_iter(x_f)
	return x_f, true 
}

out_fft_stockham_unplanned :: proc(
	algo: StockhamAlgorithm,
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	if !verfiy_power(algo, len(x)) do return

	plan, err := make_stockham_plan(len(x), C, allocator, location)
	if err != .None do return
	
	defer delete_stockham_plan(plan, allocator, location)

	return out_fft_stockham_planned(
		algo, x, plan, inverse, allocator, location
	)
}


out_fft_stockham :: proc{
	out_fft_stockham_planned,
	out_fft_stockham_unplanned
}

in_fft_stockham_planned :: proc(
	algo: StockhamAlgorithm,
	x: []$C,
	plan: Stockham_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	n := plan.n
	if len(x) != plan.n do return

	if !verfiy_power(algo, n) do return

	if n == 1 do return
	
	if inverse do conj_iter(x)

	switch algo{
		case .Radix2:
			_inner_fft_stockham_radix2(n, 1, x, plan.work, plan.twiddles, flip=false)
		case .Radix3:
			_inner_fft_stockham_radix3(n, 1, x, plan.work, plan.twiddles, flip=false)
		case .Radix4:
			_inner_fft_stockham_radix4(n, 1, x, plan.work, plan.twiddles, flip=false)
		case .Radix5:
			_inner_fft_stockham_radix5(n, 1, x, plan.work, plan.twiddles, flip=false)
		case .MixedRadix:
			_inner_fft_stockham_mixed(n, 1, x, plan.work, plan.twiddles, flip=false)
	}

	if inverse do for &val in x do val /= complex(cast(F)n, 0)
	if inverse do conj_iter(x)
	return true
}


in_fft_stockham_unplanned :: proc(
	algo: StockhamAlgorithm,
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	if !verfiy_power(algo, len(x)) do return

	plan, err := make_stockham_plan(len(x), C, allocator, location)
	if err != .None do return

	defer delete_stockham_plan(plan, allocator, location)

	return in_fft_stockham_planned(
		algo, x, plan, inverse, location
	)
}


in_fft_stockham :: proc{
	in_fft_stockham_planned,
	in_fft_stockham_unplanned
}
