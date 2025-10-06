package linalg_tests

import "core:log"
import "core:math"
import "core:testing"

@require import md "../../nuod/mdarray"
@require import nl "../../nuod/linalg"

@(test)
test_inner_product :: proc(t: ^testing.T){
	arr := md.from_range(f64, 5)
	defer md.free_mdarray(arr)

	val := nl.inner_product(arr, arr)
	norm := nl.full_vector_norm(arr)

	testing.expect_value(t, math.sqrt(val), norm)	
}


@(test)
test_outer_product :: proc(t: ^testing.T){
	arr := md.from_range(f64, 4)
	defer md.free_mdarray(arr)

	outer_arr := nl.outer_product(arr, arr)
	defer md.free_mdarray(outer_arr)

	testing.expect_value(t, outer_arr.shape, [2]int{4, 4})
	testing.expect_value(t, len(outer_arr.buffer), 16)

	expect_arr := []f64{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9}

	for i in 0..<16{
		testing.expect_value(t, outer_arr.buffer[i], expect_arr[i])
	}
}


@(test)
test_kron_product :: proc(t: ^testing.T){
	arr := md.from_range(f64, 4)
	defer md.free_mdarray(arr)

	outer_arr := nl.kron_vector_product(arr, arr)
	defer md.free_mdarray(outer_arr)

	testing.expect_value(t, outer_arr.shape, [1]int{16})
	testing.expect_value(t, len(outer_arr.buffer), 16)

	expect_arr := []f64{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9}

	for i in 0..<16{
		testing.expect_value(t, outer_arr.buffer[i], expect_arr[i])
	}
}


@(test)
test_2d_matmul :: proc(t: ^testing.T){
	matmul_2d_type_tester :: proc (t: ^testing.T, $T: typeid) {
		arr := md.reshaped_range(T, [2]int{2, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [2]int{3, 2})
		defer md.free_mdarray(arr2)

		mat := nl.matmul(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [2]int{2, 2})
		testing.expect_value(t, len(mat.buffer), 4)

		expect_arr := []T{10, 13, 28, 40}
		for i in 0..<4{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_2d_type_tester(t, f64)
	matmul_2d_type_tester(t, f32)
	matmul_2d_type_tester(t, u32)
	matmul_2d_type_tester(t, int)

	{
		arr := md.from_slice([]complex64{0, 1, 2, 3, 4, 5}, [2]int{2, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.from_slice([]complex64{0, 1, 2, 3, 4, 5}, [2]int{3, 2})
		defer md.free_mdarray(arr2)

		mat := nl.matmul(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [2]int{2, 2})
		testing.expect_value(t, len(mat.buffer), 4)

		expect_arr := []complex64{10+0i, 13+0i, 28+0i, 40+0i}
		for i in 0..<4{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}
}


@(test)
test_narrowed_2d_matmul :: proc(t: ^testing.T){
	matmul_2d_type_tester :: proc (t: ^testing.T, $T: typeid) {
		arr_ := md.reshaped_range(T, [2]int{4, 3})
		defer md.free_mdarray(arr_)

		arr := md.narrow(arr_, 0, 0, 2)

		arr2 := md.reshaped_range(T, [2]int{3, 2})
		defer md.free_mdarray(arr2)

		mat := nl.matmul(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [2]int{2, 2})
		testing.expect_value(t, len(mat.buffer), 4)

		expect_arr := []T{10, 13, 28, 40}
		for i in 0..<4{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_2d_type_tester(t, f64)
	matmul_2d_type_tester(t, f32)
	matmul_2d_type_tester(t, u32)
	matmul_2d_type_tester(t, int)

}


@(test)
test_nd_matmul :: proc(t: ^testing.T){
	matmul_3d_type_tester :: proc (t: ^testing.T, $T: typeid){
		arr := md.reshaped_range(T, [3]int{2, 2, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [3]int{2, 3, 2})
		defer md.free_mdarray(arr2)

		mat := nl.matmul(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [3]int{2, 2, 2})
		testing.expect_value(t, len(mat.buffer), 8)

		expect_arr := []T{10, 13, 28, 40, 172, 193, 244, 274}

		for i in 0..<8{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_4d_type_tester :: proc (t: ^testing.T, $T: typeid){
		arr := md.reshaped_range(T, [4]int{2, 2, 2, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [4]int{2, 2, 3, 2})
		defer md.free_mdarray(arr2)

		mat := nl.matmul(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [4]int{2, 2, 2, 2})
		testing.expect_value(t, len(mat.buffer), 16)

		expect_arr := []T{10, 13, 28, 40, 172, 193, 244, 274, 550, 589, 676, 724}

		for i in 0..<12{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}
	matmul_3d_type_tester(t, f64)
	matmul_3d_type_tester(t, f32)
	matmul_3d_type_tester(t, i32)
	matmul_3d_type_tester(t, u64)

	matmul_4d_type_tester(t, f64)
	matmul_4d_type_tester(t, f32)
	matmul_4d_type_tester(t, i32)
	matmul_4d_type_tester(t, u64)
}


@(test)
test_2d_matvec :: proc(t: ^testing.T){
	matmul_2d_type_tester :: proc (t: ^testing.T, $T: typeid) {
		arr := md.reshaped_range(T, [2]int{3, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.from_range(T, 3)
		defer md.free_mdarray(arr2)

		mat := nl.matvec(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [1]int{3})
		testing.expect_value(t, len(mat.buffer), 3)

		expect_arr := []T{5, 14, 23}
		for i in 0..<3{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_2d_type_tester(t, f64)
	matmul_2d_type_tester(t, f32)
	matmul_2d_type_tester(t, u32)
	matmul_2d_type_tester(t, int)
}

@(test)
test_nd_matvec :: proc(t: ^testing.T){
	matmul_3d_type_tester :: proc (t: ^testing.T, $T: typeid){
		arr := md.reshaped_range(T, [3]int{2, 2, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [2]int{2, 3})
		defer md.free_mdarray(arr2)

		mat := nl.matvec(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [2]int{2, 2})
		testing.expect_value(t, len(mat.buffer), 4)

		expect_arr := []T{5, 14, 86, 122}

		for i in 0..<4{
		 testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_4d_type_tester :: proc (t: ^testing.T, $T: typeid){
		arr := md.reshaped_range(T, [4]int{2, 2, 2, 3})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [3]int{2, 2, 3})
		defer md.free_mdarray(arr2)

		mat := nl.matvec(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [3]int{2, 2, 2})
		testing.expect_value(t, len(mat.buffer), 8)


		expect_arr := []T{5, 14, 86, 122, 275, 338, 572, 662}
		for i in 0..<8{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}
	matmul_3d_type_tester(t, f64)
	matmul_3d_type_tester(t, f32)
	matmul_3d_type_tester(t, i32)
	matmul_3d_type_tester(t, u64)

	matmul_4d_type_tester(t, f64)
	matmul_4d_type_tester(t, f32)
	matmul_4d_type_tester(t, i32)
	matmul_4d_type_tester(t, u64)
}


@(test)
test_narrowed_2d_matvec :: proc(t: ^testing.T){
	matmul_2d_type_tester :: proc (t: ^testing.T, $T: typeid) {
		arr_ := md.reshaped_range(T, [2]int{4, 3})
		defer md.free_mdarray(arr_)

		arr := md.narrow(arr_, 0, 0, 2)

		arr2 := md.reshaped_range(T, [1]int{3})
		defer md.free_mdarray(arr2)

		mat := nl.matvec(arr, arr2)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [1]int{2})
		testing.expect_value(t, len(mat.buffer), 2)

		expect_arr := []T{5, 14}
		for i in 0..<2{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_2d_type_tester(t, f64)
	matmul_2d_type_tester(t, f32)
	matmul_2d_type_tester(t, u32)
	matmul_2d_type_tester(t, int)

}


@(test)
test_2d_vecmat :: proc(t: ^testing.T){
	matmul_2d_type_tester :: proc (t: ^testing.T, $T: typeid) {
		arr := md.reshaped_range(T, [2]int{3, 2})
		defer md.free_mdarray(arr)
		
		arr2 := md.from_range(T, 3)
		defer md.free_mdarray(arr2)

		mat := nl.vecmat(arr2, arr)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [1]int{2})
		testing.expect_value(t, len(mat.buffer), 2)

		expect_arr := []T{10, 13}
		for i in 0..<2{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_2d_type_tester(t, f64)
	matmul_2d_type_tester(t, f32)
	matmul_2d_type_tester(t, u32)
	matmul_2d_type_tester(t, int)
}


@(test)
test_nd_vecmat :: proc(t: ^testing.T){
	matmul_3d_type_tester :: proc (t: ^testing.T, $T: typeid){
		arr := md.reshaped_range(T, [3]int{2, 3, 2})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [2]int{2, 3})
		defer md.free_mdarray(arr2)

		mat := nl.vecmat(arr2, arr)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [2]int{2, 2})
		testing.expect_value(t, len(mat.buffer), 4)

		expect_arr := []T{10, 13}

		for i in 0..<2{
		 testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}

	matmul_4d_type_tester :: proc (t: ^testing.T, $T: typeid){
		arr := md.reshaped_range(T, [4]int{2, 2, 3, 2})
		defer md.free_mdarray(arr)
		
		arr2 := md.reshaped_range(T, [3]int{2, 2, 3})
		defer md.free_mdarray(arr2)

		mat := nl.vecmat(arr2, arr)
		defer md.free_mdarray(mat)

		
		testing.expect_value(t, mat.shape, [3]int{2, 2, 2})
		testing.expect_value(t, len(mat.buffer), 8)


		expect_arr := []T{10, 13}
		for i in 0..<2{
			testing.expect_value(t, mat.buffer[i], expect_arr[i])
		}
	}
	matmul_3d_type_tester(t, f64)
	matmul_3d_type_tester(t, f32)
	matmul_3d_type_tester(t, i32)
	matmul_3d_type_tester(t, u64)

	matmul_4d_type_tester(t, f64)
	matmul_4d_type_tester(t, f32)
	matmul_4d_type_tester(t, i32)
	matmul_4d_type_tester(t, u64)
}

