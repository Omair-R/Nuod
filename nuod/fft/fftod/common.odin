package fftod

import "core:math"
import "base:runtime"
import "base:intrinsics"
import "core:math/cmplx"


rect_complex :: proc{cmplx.rect_complex128, cmplx.rect_complex64, cmplx.acos_complex32}


conj_iter ::proc (x : $T/[]$E) where intrinsics.type_is_complex(E) {
	for &val in x {
		val = conj(val)
	}
}

// k^floor(log(k, 2^31-1))
is_power5 :: proc(n: int) -> bool {
	when size_of(n) == size_of(i64) do return n > 0 && 1220703125 % n == 0 
	when size_of(n) == size_of(i32) do return n > 0 && 7450580596923828125 % n == 0 
	return false
}


is_power4 :: proc(n: int) -> bool {
	return n > 0 && ((n & (n - 1)) == 0) &&
        	(n & 0xAAAAAAAA) == 0
}

is_power3 :: proc(n: int) -> bool {
	when size_of(n) == size_of(i64) do return n > 0 && 1162261467 % n == 0 
	when size_of(n) == size_of(i32) do return n > 0 && 4052555153018976267 % n == 0 
	return false
}

is_power2 :: proc(n: int) -> bool {
	return (n > 0) && ((n & (n - 1)) == 0)
}

is_factorizable :: proc(n: int) -> bool {
	n := n

	if n < 0 do return false

	for n > 1{
		if n % 2 == 0 do n /=2
		else if n % 3 == 0 do n /=3
		else if n % 5 == 0 do n /=5
		else do return false
	}
	return true 
}
