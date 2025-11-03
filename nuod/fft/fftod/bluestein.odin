package fftod


import "core:log"
import "base:runtime"
import "base:intrinsics"
import "core:math"


Bluestein_Plan :: struct($C:typeid)
where intrinsics.type_is_complex(C)
{
	n: int,
	nl: int,
	work: []C,
	work_a: []C,
	work_b: []C,
	work_w: []C,
	twiddles: []C,
}


make_bluestein_plan :: proc(
	n: int,
	$C: typeid,
	allocator := context.allocator,
	location := #caller_location,
) -> (
	plan: Bluestein_Plan(C),
	err: runtime.Allocator_Error,
) where intrinsics.type_is_complex(C) #optional_allocator_error {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	nl := int(math.pow(2, math.ceil(math.log2(2*f64(n)))))

	work := make([]C, nl, allocator, location) or_return
	work_a := make([]C, nl, allocator, location) or_return
	work_b := make([]C, nl, allocator, location) or_return
	work_w := make([]C, n, allocator, location) or_return
	twiddles := make([]C, nl/2, allocator, location) or_return

	theta0 :F = math.TAU/F(nl)

	for i in 0..<(nl/2){
		twiddles[i] = rect_complex(1.0, -F(i+1)*theta0)
	}

	return Bluestein_Plan(C){
		n,
		nl,
		work,
		work_a,
		work_b,
		work_w,
		twiddles,
	}, .None
}


delete_bluestein_plan :: proc(
	plan: Bluestein_Plan($C),
	allocator:=context.allocator,
	location := #caller_location,
) -> runtime.Allocator_Error {
	err := delete(plan.work, allocator, location)
	if err != .None do return err
	err = delete(plan.work_a, allocator, location)
	if err != .None do return err
	err = delete(plan.work_b, allocator, location)
	if err != .None do return err
	err = delete(plan.work_w, allocator, location)
	if err != .None do return err
	return delete(plan.twiddles, allocator, location)
}


@private
_inner_fft_bluestein :: proc (
	n:int,
	nl: int,
	x : []$C,
	work:[]C,
	work_a:[]C,
	work_b:[]C,
	work_w:[]C,
	twiddles:[]C,
	inverse:=false,
) where intrinsics.type_is_complex(C) {
	
	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16

	w_sign :F = 1 if inverse else -1

	for i in 1..<n{
		i_f := F(i)
		theta := math.PI/F(n) * i_f*i_f
		work_w[i] = rect_complex(1,  w_sign * theta)
		work_b[i] = conj(work_w[i])
		work_b[nl-i] = work_b[i]
	}
	work_w[0] = complex(1, 0)
	work_b[0] = complex(1, 0)

	for i in 0..<n {
		work_a[i] = x[i] * work_w[i]
	}

	_inner_fft_stockham_radix2(nl, 1, work_a, work, twiddles, false)
	_inner_fft_stockham_radix2(nl, 1, work_b, work, twiddles, false)

	for i in 0..<nl{
		work_b[i] = work_a[i] * work_b[i]
	}

	conj_iter(work_b)
	_inner_fft_stockham_radix2(nl, 1, work_b, work, twiddles, false)
	conj_iter(work_b)
	for i in 0..<n{
		x[i] = work_w[i] * work_b[i]/(complex(F(nl), 0)) //scale for the inverse
	}

	if inverse do for i in 0..<n{
		x[i] /= (complex(F(n), 0)) 
	}
}


out_fft_bluestein_planned :: proc(
	x: []$C,
	plan: Bluestein_Plan(C),
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	if len(x) != plan.n do return
	
	x_f = make([]C, plan.n, allocator, location)
	copy(x_f, x)
	
	if plan.n == 1 do return x_f, true

	_inner_fft_bluestein(
		plan.n,
		plan.nl,
		x_f,
		plan.work,
		plan.work_a,
		plan.work_b,
		plan.work_w,
		plan.twiddles,
		inverse=inverse
	)

	return x_f, true
}

out_fft_bluestein_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> ( x_f: []C, ok:bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	plan, err := make_bluestein_plan(len(x), C, allocator, location)
	if err != .None do return

	defer delete_bluestein_plan(plan, allocator, location)

	return out_fft_bluestein_planned(
		x, plan, inverse, allocator, location
	)
}


out_fft_bluestein :: proc{
	out_fft_bluestein_planned,
	out_fft_bluestein_unplanned
}

in_fft_bluestein_planned :: proc(
	x: []$C,
	plan: Bluestein_Plan(C),
	inverse:=false,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	if len(x) != plan.n do return
	if plan.n == 1 do return
	
	_inner_fft_bluestein(
		plan.n,
		plan.nl,
		x,
		plan.work,
		plan.work_a,
		plan.work_b,
		plan.work_w,
		plan.twiddles,
		inverse=inverse
	)

	return true
}


in_fft_bluestein_unplanned :: proc(
	x: []$C,
	inverse:=false,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	plan, err := make_bluestein_plan(len(x), C, allocator, location)
	if err != .None do return

	defer delete_bluestein_plan(plan, allocator, location)

	in_fft_bluestein_planned(
		x, plan, inverse, location
	)

	return true
}


in_fft_bluestein :: proc{
	in_fft_bluestein_planned,
	in_fft_bluestein_unplanned
}
