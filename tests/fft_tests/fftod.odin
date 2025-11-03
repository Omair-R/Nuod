package fft_tests


import "base:runtime"
import "core:log"
import "base:intrinsics"
import "core:testing"
import "core:math"
import "core:math/cmplx"
import "core:math/rand"
import fftod "../../nuod/fft/fftod"


Transform:: enum {
	Radix2,
	Radix3,
	Radix4,
	Radix5,
	MixedRadix,
	Bluestein, 
}

inner_test_unplanned :: proc(
	t: ^testing.T,
	N: int,
	$C: typeid,
	transform : Transform,
)
	where intrinsics.type_is_complex(C)
{
	when C == complex128{
		EPS := EPS_F64
	} else{
		EPS := EPS_F32
	}

	x := make([]C, N)		
	defer delete(x)

	for i in 0..<N{
		x[i] = complex(rand.float32(), rand.float32())
	}

	f:[]C

	switch transform {
		case .Radix2:
			f = fftod.out_fft_stockham_radix2(x, false)
		case .Radix3:
			f = fftod.out_fft_stockham_radix3(x, false)
		case .Radix4:
			f = fftod.out_fft_stockham_radix4(x, false)
		case .Radix5:
			f = fftod.out_fft_stockham_radix5(x, false)
		case .MixedRadix:
			f = fftod.out_fft_stockham_mixed(x, false)
		case .Bluestein:
			f = fftod.out_fft_bluestein(x, false)

	}

	defer delete(f)
	
	f_ := naive_dft(x)
	defer delete(f_)

	testing.expect_value(t, len(f), len(x))

	close := is_all_close_c(f, f_, EPS)
	testing.expect(t, close)

	x_ : []C

	switch transform {
		case .Radix2:
			x_ = fftod.out_fft_stockham_radix2(f, true)
		case .Radix3:
			x_ = fftod.out_fft_stockham_radix3(f, true)
		case .Radix4:
			x_ = fftod.out_fft_stockham_radix4(f, true)
		case .Radix5:
			x_ = fftod.out_fft_stockham_radix5(f, true)
		case .MixedRadix:
			x_ = fftod.out_fft_stockham_mixed(f, true)
		case .Bluestein:
			x_ = fftod.out_fft_bluestein(f, true)
	}

	defer delete(x_)

	close = is_all_close(x, x_, EPS)
	testing.expect(t, close)
}


@test
test_radix2 :: proc(t : ^testing.T){
	inner_test_unplanned(t, 1, complex64, .Radix2)
	inner_test_unplanned(t, 16, complex64, .Radix2)
	inner_test_unplanned(t, 32, complex64, .Radix2)
	inner_test_unplanned(t, 64, complex64, .Radix2)
	inner_test_unplanned(t, 128, complex64, .Radix2)
	inner_test_unplanned(t, 256, complex64, .Radix2)
	inner_test_unplanned(t, 512, complex64, .Radix2)

	inner_test_unplanned(t, 16, complex128, .Radix2)
	inner_test_unplanned(t, 32, complex128, .Radix2)
	inner_test_unplanned(t, 64, complex128, .Radix2)
	inner_test_unplanned(t, 128, complex128, .Radix2)
	inner_test_unplanned(t, 256, complex128, .Radix2)
	inner_test_unplanned(t, 512, complex128, .Radix2)
	inner_test_unplanned(t, 1024, complex128, .Radix2)
}


@test
test_radix3 :: proc(t : ^testing.T){
	inner_test_unplanned(t, 1, complex64, .Radix3)
	inner_test_unplanned(t, 9, complex64, .Radix3)
	inner_test_unplanned(t, 27, complex64, .Radix3)
	inner_test_unplanned(t, 81, complex64, .Radix3)
	inner_test_unplanned(t, 243, complex64, .Radix3)

	inner_test_unplanned(t, 9, complex128, .Radix3)
	inner_test_unplanned(t, 27, complex128, .Radix3)
	inner_test_unplanned(t, 81, complex128, .Radix3)
	inner_test_unplanned(t, 243, complex128, .Radix3)
	inner_test_unplanned(t, 729, complex128, .Radix3)
	inner_test_unplanned(t, 2187, complex128, .Radix3)
}


@test
test_radix4 :: proc(t : ^testing.T){
	inner_test_unplanned(t, 1, complex64, .Radix4)
	inner_test_unplanned(t, 16, complex64, .Radix4)
	inner_test_unplanned(t, 64, complex64, .Radix4)
	inner_test_unplanned(t, 256, complex64, .Radix4)

	inner_test_unplanned(t, 16, complex128, .Radix4)
	inner_test_unplanned(t, 64, complex128, .Radix4)
	inner_test_unplanned(t, 256, complex128, .Radix4)
	inner_test_unplanned(t, 1024, complex128, .Radix4)
	inner_test_unplanned(t, 4096, complex128, .Radix4)
}


@test
test_radix5 :: proc(t : ^testing.T){
	inner_test_unplanned(t, 1, complex64, .Radix5)
	inner_test_unplanned(t, 25, complex64, .Radix5)
	inner_test_unplanned(t, 125, complex64, .Radix5)
	inner_test_unplanned(t, 625, complex64, .Radix5)

	inner_test_unplanned(t, 25, complex128, .Radix5)
	inner_test_unplanned(t, 125, complex128, .Radix5)
	inner_test_unplanned(t, 625, complex128, .Radix5)
}


@test
test_mixed :: proc(t : ^testing.T){
	inner_test_unplanned(t, 1, complex64, .MixedRadix)
	inner_test_unplanned(t, 9, complex64,  .MixedRadix)
	inner_test_unplanned(t, 16, complex64, .MixedRadix)
	inner_test_unplanned(t, 25, complex64, .MixedRadix)
	inner_test_unplanned(t, 2*3*5, complex64,  .MixedRadix)
	inner_test_unplanned(t, 2*2*5, complex64, .MixedRadix)
	inner_test_unplanned(t, 3*3*3*3, complex64, .MixedRadix)

	inner_test_unplanned(t, 9, complex128,  .MixedRadix)
	inner_test_unplanned(t, 16, complex128, .MixedRadix)
	inner_test_unplanned(t, 25, complex128, .MixedRadix)
	inner_test_unplanned(t, 3*5*3*2, complex128, .MixedRadix)
	inner_test_unplanned(t, 5*5*5*5, complex128, .MixedRadix)
	inner_test_unplanned(t, 2*2*2*3, complex128, .MixedRadix)
}


@test
test_bluestein :: proc(t : ^testing.T){
	inner_test_unplanned(t, 1, complex64, .Bluestein)
	inner_test_unplanned(t, 16, complex64, .Bluestein)
	inner_test_unplanned(t, 25, complex64, .Bluestein)
	inner_test_unplanned(t, 27, complex64, .Bluestein)
	inner_test_unplanned(t, 32, complex64, .Bluestein)
	inner_test_unplanned(t, 41, complex64, .Bluestein)

	
	inner_test_unplanned(t, 16, complex128, .Bluestein)
	inner_test_unplanned(t, 25, complex128, .Bluestein)
	inner_test_unplanned(t, 27, complex128, .Bluestein)
	inner_test_unplanned(t, 32, complex128, .Bluestein)
	inner_test_unplanned(t, 41, complex128, .Bluestein)
}

inner_test_high_level :: proc(
	t: ^testing.T,
	N: int,
	$C: typeid,
)
	where intrinsics.type_is_complex(C)
{
	when C == complex128{
		EPS := EPS_F64
	} else{
		EPS := EPS_F32
	}

	x := make([]C, N)		
	defer delete(x)

	for i in 0..<N{
		x[i] = complex(rand.float32(), rand.float32())
	}

	f := fftod.out_fft(x)
	defer delete(f)
	
	f_ := naive_dft(x)
	defer delete(f_)

	testing.expect_value(t, len(f), len(x))

	close := is_all_close_c(f, f_, EPS)
	testing.expect(t, close)

	x_ := fftod.out_ifft(f)
	defer delete(x_)

	close = is_all_close(x, x_, EPS)
	testing.expect(t, close)
}


@test
test_highlevel_out_fft :: proc(t : ^testing.T){
	inner_test_high_level(t, 1, complex64)
	inner_test_high_level(t, 16, complex64)
	inner_test_high_level(t, 25, complex64)
	inner_test_high_level(t, 27, complex64)
	inner_test_high_level(t, 32, complex64)
	inner_test_high_level(t, 41, complex64)

	
	inner_test_high_level(t, 16, complex128)
	inner_test_high_level(t, 25, complex128)
	inner_test_high_level(t, 27, complex128)
	inner_test_high_level(t, 32, complex128)
	inner_test_high_level(t, 41, complex128)
}


inner_test_unplanned_in :: proc(
	t: ^testing.T,
	N: int,
	$C: typeid,
	transform : Transform,
)
	where intrinsics.type_is_complex(C)
{
	when C == complex128{
		EPS := EPS_F64
	} else{
		EPS := EPS_F32
	}

	x := make([]C, N)		
	defer delete(x)
	x_ := make([]C, N)		
	defer delete(x_)

	for i in 0..<N{
		x[i] = complex(rand.float32(), rand.float32())
	}

	copy(x_, x)


	switch transform {
		case .Radix2:
			fftod.in_fft_stockham_radix2(x_, false)
		case .Radix3:
			fftod.in_fft_stockham_radix3(x_, false)
		case .Radix4:
			fftod.in_fft_stockham_radix4(x_, false)
		case .Radix5:
			fftod.in_fft_stockham_radix5(x_, false)
		case .MixedRadix:
			fftod.in_fft_stockham_mixed(x_, false)
		case .Bluestein:
			fftod.in_fft_bluestein(x_, false)

	}
	
	f_ := naive_dft(x)
	defer delete(f_)

	testing.expect_value(t, len(x_), len(x))

	close := is_all_close_c(x_, f_, EPS)
	testing.expect(t, close)

	switch transform {
		case .Radix2:
			fftod.in_fft_stockham_radix2(x_, true)
		case .Radix3:
			fftod.in_fft_stockham_radix3(x_, true)
		case .Radix4:
			fftod.in_fft_stockham_radix4(x_, true)
		case .Radix5:
			fftod.in_fft_stockham_radix5(x_, true)
		case .MixedRadix:
			fftod.in_fft_stockham_mixed( x_, true)
		case .Bluestein:
			fftod.in_fft_bluestein(x_, true)
	}

	close = is_all_close(x, x_, EPS)
	testing.expect(t, close)
}



@test
test_radix2_in :: proc(t : ^testing.T){
	inner_test_unplanned_in(t, 1, complex64, .Radix2)
	inner_test_unplanned_in(t, 16, complex64, .Radix2)
	inner_test_unplanned_in(t, 32, complex64, .Radix2)
	inner_test_unplanned_in(t, 64, complex64, .Radix2)
	inner_test_unplanned_in(t, 128, complex64, .Radix2)
	inner_test_unplanned_in(t, 256, complex64, .Radix2)
	inner_test_unplanned_in(t, 512, complex64, .Radix2)

	inner_test_unplanned_in(t, 16, complex128, .Radix2)
	inner_test_unplanned_in(t, 32, complex128, .Radix2)
	inner_test_unplanned_in(t, 64, complex128, .Radix2)
	inner_test_unplanned_in(t, 128, complex128, .Radix2)
	inner_test_unplanned_in(t, 256, complex128, .Radix2)
	inner_test_unplanned_in(t, 512, complex128, .Radix2)
	inner_test_unplanned_in(t, 1024, complex128, .Radix2)
}


@test
test_radix3_in :: proc(t : ^testing.T){
	inner_test_unplanned_in(t, 1, complex64, .Radix3)
	inner_test_unplanned_in(t, 9, complex64, .Radix3)
	inner_test_unplanned_in(t, 27, complex64, .Radix3)
	inner_test_unplanned_in(t, 81, complex64, .Radix3)
	inner_test_unplanned_in(t, 243, complex64, .Radix3)

	inner_test_unplanned_in(t, 9, complex128, .Radix3)
	inner_test_unplanned_in(t, 27, complex128, .Radix3)
	inner_test_unplanned_in(t, 81, complex128, .Radix3)
	inner_test_unplanned_in(t, 243, complex128, .Radix3)
	inner_test_unplanned_in(t, 729, complex128, .Radix3)
	inner_test_unplanned_in(t, 2187, complex128, .Radix3)
}


@test
test_radix4_in :: proc(t : ^testing.T){
	inner_test_unplanned_in(t, 1, complex64, .Radix4)
	inner_test_unplanned_in(t, 16, complex64, .Radix4)
	inner_test_unplanned_in(t, 64, complex64, .Radix4)
	inner_test_unplanned_in(t, 256, complex64, .Radix4)

	inner_test_unplanned_in(t, 16, complex128, .Radix4)
	inner_test_unplanned_in(t, 64, complex128, .Radix4)
	inner_test_unplanned_in(t, 256, complex128, .Radix4)
	inner_test_unplanned_in(t, 1024, complex128, .Radix4)
	inner_test_unplanned_in(t, 4096, complex128, .Radix4)
}


@test
test_radix5_in :: proc(t : ^testing.T){
	inner_test_unplanned_in(t, 1, complex64, .Radix5)
	inner_test_unplanned_in(t, 25, complex64, .Radix5)
	inner_test_unplanned_in(t, 125, complex64, .Radix5)
	inner_test_unplanned_in(t, 625, complex64, .Radix5)

	inner_test_unplanned_in(t, 25, complex128, .Radix5)
	inner_test_unplanned_in(t, 125, complex128, .Radix5)
	inner_test_unplanned_in(t, 625, complex128, .Radix5)
}


@test
test_mixed_in :: proc(t : ^testing.T){
	inner_test_unplanned_in(t, 1, complex64, .MixedRadix)
	inner_test_unplanned_in(t, 9, complex64,  .MixedRadix)
	inner_test_unplanned_in(t, 16, complex64, .MixedRadix)
	inner_test_unplanned_in(t, 25, complex64, .MixedRadix)
	inner_test_unplanned_in(t, 2*3*5, complex64,  .MixedRadix)
	inner_test_unplanned_in(t, 2*2*5, complex64, .MixedRadix)
	inner_test_unplanned_in(t, 3*3*3*3, complex64, .MixedRadix)

	inner_test_unplanned_in(t, 9, complex128,  .MixedRadix)
	inner_test_unplanned_in(t, 16, complex128, .MixedRadix)
	inner_test_unplanned_in(t, 25, complex128, .MixedRadix)
	inner_test_unplanned_in(t, 3*5*3*2, complex128, .MixedRadix)
	inner_test_unplanned_in(t, 5*5*5*5, complex128, .MixedRadix)
	inner_test_unplanned_in(t, 2*2*2*3, complex128, .MixedRadix)
}


@test
test_bluestein_in :: proc(t : ^testing.T){
	inner_test_unplanned_in(t, 1, complex64, .Bluestein)
	inner_test_unplanned_in(t, 16, complex64, .Bluestein)
	inner_test_unplanned_in(t, 25, complex64, .Bluestein)
	inner_test_unplanned_in(t, 27, complex64, .Bluestein)
	inner_test_unplanned_in(t, 32, complex64, .Bluestein)
	inner_test_unplanned_in(t, 41, complex64, .Bluestein)

	
	inner_test_unplanned_in(t, 16, complex128, .Bluestein)
	inner_test_unplanned_in(t, 25, complex128, .Bluestein)
	inner_test_unplanned_in(t, 27, complex128, .Bluestein)
	inner_test_unplanned_in(t, 32, complex128, .Bluestein)
	inner_test_unplanned_in(t, 41, complex128, .Bluestein)
}

inner_test_high_level_in :: proc(
	t: ^testing.T,
	N: int,
	$C: typeid,
)
	where intrinsics.type_is_complex(C)
{
	when C == complex128{
		EPS := EPS_F64
	} else{
		EPS := EPS_F32
	}

	x := make([]C, N)		
	defer delete(x)
	x_ := make([]C, N)		
	defer delete(x_)

	for i in 0..<N{
		x[i] = complex(rand.float32(), rand.float32())
	}
	copy(x_, x)

	fftod.in_fft(x_)
	
	f_ := naive_dft(x)
	defer delete(f_)

	testing.expect_value(t, len(x_), len(x))

	close := is_all_close_c(x_, f_, EPS)
	testing.expect(t, close)

	fftod.in_ifft(x_)

	close = is_all_close(x, x_, EPS)
	testing.expect(t, close)
}


@test
test_highlevel_out_fft_in :: proc(t : ^testing.T){
	inner_test_high_level_in(t, 1, complex64)
	inner_test_high_level_in(t, 16, complex64)
	inner_test_high_level_in(t, 25, complex64)
	inner_test_high_level_in(t, 27, complex64)
	inner_test_high_level_in(t, 32, complex64)
	inner_test_high_level_in(t, 41, complex64)

	
	inner_test_high_level_in(t, 16, complex128)
	inner_test_high_level_in(t, 25, complex128)
	inner_test_high_level_in(t, 27, complex128)
	inner_test_high_level_in(t, 32, complex128)
	inner_test_high_level_in(t, 41, complex128)
}


