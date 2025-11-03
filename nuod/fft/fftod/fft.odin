package fftod

import "base:intrinsics"


out_fft :: proc(
	x: []$C,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (result: []C, ok: bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	n:= len(x)
	if is_power4(n) {
		return out_fft_stockham_radix4(x, inverse=false, allocator=allocator, location=location)
	} else if is_power2(n) {
		return out_fft_stockham_radix2(x, inverse=false, allocator=allocator, location=location)
	} else if is_power3(n) {
		return out_fft_stockham_radix3(x, inverse=false, allocator=allocator, location=location)
	} else if is_power5(n) {
		return out_fft_stockham_radix5(x, inverse=false, allocator=allocator, location=location)
	} else if is_factorizable(n) {
		return out_fft_stockham_mixed(x, inverse=false,  allocator=allocator, location=location)
	} else {
		return out_fft_bluestein(x, inverse=false,       allocator=allocator, location=location)
	}
}


out_ifft :: proc(
	x_f: []$C,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (result: []C, ok: bool)
	where intrinsics.type_is_complex(C) #optional_ok
{
	n:= len(x_f)
	if is_power4(n) {
		return out_fft_stockham_radix4(x_f, inverse=true, allocator=allocator, location=location)
	} else if is_power2(n) {
		return out_fft_stockham_radix2(x_f, inverse=true, allocator=allocator, location=location)
	} else if is_power3(n) {
		return out_fft_stockham_radix3(x_f, inverse=true, allocator=allocator, location=location)
	} else if is_power5(n) {
		return out_fft_stockham_radix5(x_f, inverse=true, allocator=allocator, location=location)
	} else if is_factorizable(n) {
		return out_fft_stockham_mixed(x_f, inverse=true,  allocator=allocator, location=location)
	} else {
		return out_fft_bluestein(x_f, inverse=true,       allocator=allocator, location=location)
	}
}


in_fft :: proc(
	x: []$C,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	n:= len(x)
	if is_power4(n) {
		return in_fft_stockham_radix4(x, inverse=false, allocator=allocator, location=location)
	} else if is_power2(n) {
		return in_fft_stockham_radix2(x, inverse=false, allocator=allocator, location=location)
	} else if is_power3(n) {
		return in_fft_stockham_radix3(x, inverse=false, allocator=allocator, location=location)
	} else if is_power5(n) {
		return in_fft_stockham_radix5(x, inverse=false, allocator=allocator, location=location)
	} else if is_factorizable(n) {
		return in_fft_stockham_mixed(x, inverse=false,  allocator=allocator, location=location)
	} else {
		return in_fft_bluestein(x, inverse=false,       allocator=allocator, location=location)
	}
}


in_ifft :: proc(
	x: []$C,
	allocator:=context.allocator,
	location:=#caller_location,
) -> (ok: bool)
	where intrinsics.type_is_complex(C)
{
	n:= len(x)
	if is_power4(n) {
		return in_fft_stockham_radix4(x, inverse=true, allocator=allocator, location=location)
	} else if is_power2(n) {
		return in_fft_stockham_radix2(x, inverse=true, allocator=allocator, location=location)
	} else if is_power3(n) {
		return in_fft_stockham_radix3(x, inverse=true, allocator=allocator, location=location)
	} else if is_power5(n) {
		return in_fft_stockham_radix5(x, inverse=true, allocator=allocator, location=location)
	} else if is_factorizable(n) {
		return in_fft_stockham_mixed(x, inverse=true,  allocator=allocator, location=location)
	} else {
		return in_fft_bluestein(x, inverse=true,       allocator=allocator, location=location)
	}
}
