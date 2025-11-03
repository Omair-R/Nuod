package fft_tests


import "core:testing"
import fftod "../../nuod/fft/fftod"
import "base:intrinsics"
import "core:math"
import "core:math/cmplx"


EPS_F32 :: 1e-2
EPS_F64 :: 1e-5


naive_dft :: proc(x: []$C) -> []C where intrinsics.type_is_complex(C) {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16
	N := len(x)

	f := make([]C, N)

	for k in 0..<N{
		sum : C
		for n in 0..<N{
			theta : F = math.TAU * (F(k)/F(N)) * F(n)
			sum += x[n] * fftod.rect_complex(1, -theta)
		}
		f[k] = sum
	}

	return f
}


naive_idft :: proc(f: []$C) -> []C {

	when C == complex128 do F :: f64
	else when C == complex64 do F :: f32
	else do F :: f16
	N := len(f)

	x := make([]C, N)

	for k in 0..<N{
		sum : C
		for n in 0..<N{
			theta : F = math.TAU * (F(k)/F(N)) * F(n)
			sum += f[n] * fftod.rect_complex(1, theta)
		}
		x[k] = sum/complex(f64(F(x)), 0)
	}

	return x
}


is_close:: proc(x, y: $F, eps:=1e-5) -> bool where intrinsics.type_is_float(F) {
	return abs(x - y) < F(eps) 
}


is_all_close_c :: proc(x, y: $T/[]$E, eps:=1e-5) -> (bool) where intrinsics.type_is_complex(E) {
	if len(x) != len(y) do return false
	
	for i in 0..<len(x){
		if !is_close(real(x[i]), real(y[i]), eps) do return false
		if !is_close(imag(x[i]), imag(y[i]), eps) do return false
	}
	return true
}


is_all_close_f :: proc(x, y: $T/[]$E, eps:=1e-5) -> (bool) where intrinsics.type_is_float(E) {
	if len(x) != len(y) do return false
	
	for i in 0..<len(x){
		if !is_close(x[i], y[i], eps) do return false
	}
	return true
}


is_all_close :: proc{is_all_close_c, is_all_close_f}


@test
test_power2 :: proc(t: ^testing.T){
	testing.expect(t, fftod.is_power2(64))
	testing.expect(t, fftod.is_power2(32))
	testing.expect(t, fftod.is_power2(16))
	testing.expect(t, fftod.is_power2(256))

	testing.expect_value(t, fftod.is_power2(257), false)
	testing.expect_value(t, fftod.is_power2(62), false)
	testing.expect_value(t, fftod.is_power2(21), false)
}


@test
test_power3 :: proc(t: ^testing.T){
	testing.expect(t, fftod.is_power3(3))
	testing.expect(t, fftod.is_power3(9))
	testing.expect(t, fftod.is_power3(27))
	testing.expect(t, fftod.is_power3(81))

	testing.expect_value(t, fftod.is_power3(257), false)
	testing.expect_value(t, fftod.is_power3(62), false)
	testing.expect_value(t, fftod.is_power3(21), false)
}


@test
test_power4 :: proc(t: ^testing.T){
	testing.expect(t, fftod.is_power4(4))
	testing.expect(t, fftod.is_power4(16))
	testing.expect(t, fftod.is_power4(64))
	testing.expect(t, fftod.is_power4(256))

	testing.expect_value(t, fftod.is_power4(257), false)
	testing.expect_value(t, fftod.is_power4(62), false)
	testing.expect_value(t, fftod.is_power4(21), false)
}


@test
test_power5 :: proc(t: ^testing.T){
	testing.expect(t, fftod.is_power5(5))
	testing.expect(t, fftod.is_power5(25))
	testing.expect(t, fftod.is_power5(125))
	testing.expect(t, fftod.is_power5(625))

	testing.expect_value(t, fftod.is_power5(257), false)
	testing.expect_value(t, fftod.is_power5(62), false)
	testing.expect_value(t, fftod.is_power5(21), false)
}


@test
test_factorizable :: proc(t: ^testing.T){
	testing.expect(t, fftod.is_factorizable(2))
	testing.expect(t, fftod.is_factorizable(3))
	testing.expect(t, fftod.is_factorizable(5))
	testing.expect(t, fftod.is_factorizable(2*3))
	testing.expect(t, fftod.is_factorizable(2*5))
	testing.expect(t, fftod.is_factorizable(3*5))
	testing.expect(t, fftod.is_factorizable(2*3*5))
	testing.expect(t, fftod.is_factorizable(2*2*3*5))
	testing.expect(t, fftod.is_factorizable(2*3*3*5))
	testing.expect(t, fftod.is_factorizable(3*3*3*3))
	testing.expect(t, fftod.is_factorizable(5*5*5))

	testing.expect_value(t, fftod.is_factorizable(2*3*3*5*7), false)
	testing.expect_value(t, fftod.is_factorizable(2*3*3*11*7), false)
	testing.expect_value(t, fftod.is_factorizable(2*11*5), false)
	testing.expect_value(t, fftod.is_factorizable(7), false)
}
