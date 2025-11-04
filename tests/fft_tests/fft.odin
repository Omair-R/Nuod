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

