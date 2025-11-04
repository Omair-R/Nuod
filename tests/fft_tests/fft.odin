package fft_tests

import "core:log"
import "core:math/rand"
import md "../../nuod/mdarray"
import nfft "../../nuod/fft"
import "core:testing"


@test
test_fft :: proc(t: ^testing.T){

	EPS := EPS_F64
	{
		arr := md.zeros(complex128, [2]int{2, 4})
		defer md.free_mdarray(arr)

		
		for i in 0..<8{
			arr.buffer[i] = complex(rand.float64(), rand.float64())
		}


		f_arr := nfft.fft(arr) 
		defer md.free_mdarray(f_arr)

		arr_ := nfft.ifft(f_arr) 
		defer md.free_mdarray(arr_)

		close := is_all_close(arr_.buffer, arr.buffer, EPS)
		testing.expect(t, close)

		{
			f_ := naive_dft(arr.buffer[:4])
			defer delete(f_)

			close := is_all_close(f_arr.buffer[:4], f_, EPS)
			testing.expect(t, close)
		}

		{
			f_ := naive_dft(arr.buffer[4:])
			defer delete(f_)

			close := is_all_close(f_arr.buffer[4:], f_, EPS)
			testing.expect(t, close)
		}
	}
	{
		arr := md.zeros(complex128, [2]int{4, 2})
		defer md.free_mdarray(arr)

		
		for i in 0..<8{
			arr.buffer[i] = complex(rand.float64(), rand.float64())
		}


		f_arr := nfft.fft(arr, axis=0) 
		defer md.free_mdarray(f_arr)

		arr_ := nfft.ifft(f_arr, axis=0) 
		defer md.free_mdarray(arr_)

		close := is_all_close(arr_.buffer, arr.buffer, EPS)
		testing.expect(t, close)

		{
			x_ := make([]complex128, 4)
			defer delete(x_)

			md.extract_linear_array(arr, x_, 0, 0)
			f_ := naive_dft(x_)
			defer delete(f_)

			md.extract_linear_array(f_arr, x_, 0, 0)
			close := is_all_close(f_, x_, EPS)
			testing.expect(t, close)
		}

	}
}


// TODO: this is only a tempurary test, Need a better test against naive DFT instea.
@test
test_fft2d :: proc(t: ^testing.T){

	{
		arr := md.from_slice([]complex128{
			complex(1, 0),complex(2, 0),complex(3, 0),complex(4, 0),
			complex(5, 0),complex(6, 0),complex(7, 0),complex(8, 0) 
		}, [2]int{2, 4})
		defer md.free_mdarray(arr)

		
		arr_f_ := md.from_slice([]complex128{
			complex(36, 0),complex(-4, 4),complex(-4, 0),complex(-4, -4),
			complex(-16, 0),complex(0, 0),complex(0, 0),complex(0, 0) 
		}, [2]int{2, 4})
		defer md.free_mdarray(arr_f_)

		arr_f := nfft.fft2d(arr)
		defer md.free_mdarray(arr_f)

		arr_ := nfft.fft2d(arr_f, true)
		defer md.free_mdarray(arr_)


		close := md.all_close(arr_f, arr_f_)
		testing.expect(t, close)

		close = md.all_close(arr, arr_)
		testing.expect(t, close)
	}
}


@test
test_fftnd :: proc(t: ^testing.T){

	{
		arr := md.from_slice([]complex128{
			complex(1, 0),complex(2, 0),complex(3, 0),complex(4, 0),
			complex(5, 0),complex(6, 0),complex(7, 0),complex(8, 0),
			complex(2, 0),complex(1, 0),complex(2, 0),complex(4, 0),
			complex(5, 0),complex(6, 0),complex(7, 0),complex(8, 0) 
		}, [3]int{2, 2, 4})
		defer md.free_mdarray(arr)

		
		arr_f_ := md.from_slice([]complex128{
			complex(71, 0),complex(-6, 9),complex(-7, 0),complex(-6, -9),
			complex(-33, 0),complex(2, 1),complex(1, 0),complex(2, -1),
			complex(1, 0),complex(-2, -1),complex(-1, 0),complex(-2, 1),
			complex(1, 0),complex(-2, -1),complex(-1, 0),complex(-2, 1) 
		}, [3]int{2, 2, 4})
		defer md.free_mdarray(arr_f_)

		arr_f := nfft.fftnd(arr)
		defer md.free_mdarray(arr_f)

		arr_ := nfft.fftnd(arr_f, true)
		defer md.free_mdarray(arr_)

		close := md.all_close(arr_f, arr_f_)
		testing.expect(t, close)

		close = md.all_close(arr, arr_)
		testing.expect(t, close)

	}
}
