package linalg


import "core:slice"
import "base:intrinsics"
import md "../mdarray"
import "../logging"
import lapacke "../lapacke"

det_matrix :: proc(	
	a: md.MdArray($T, 2),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	de: T, 
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) #optional_ok {

	validate_open_blas(a, allocator, location) or_return

	n:= a.shape[0]
	if n != a.shape[1] {
		logging.error(
			.ArguementError,
			"the determinant can only be computed for nonsymteric square matrices or stacks of matrices.",
			location=location,
		)
	}

	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	return lapacke_det_matrix(a_, allocator, location) 	
}


det_tensor :: proc(	
	$Nd: int, 
	a: md.MdArray($T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	de: md.MdArray(T, Nd-2), 
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 #optional_ok {

	validate_open_blas(a, allocator, location) or_return

	n:= a.shape[Nd-1]
	if n != a.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"the determinant can only be computed for nonsymteric square matrices or stacks of matrices.",
			location=location,
		)
	}

	
	a_ := md.copy_array(a, allocator, location) or_return
	defer md.free_mdarray(a_)

	return lapacke_det(3, a_, allocator, location) 	
}


det :: proc { det_matrix, det_tensor}


slog_det_matrix :: proc(	
	a: md.MdArray($T, 2),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	sign: int,
	slog_de: T, 
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {

	de := det_matrix(a, allocator, location) or_return

	sign = math.sign(de)

	slog_de = abs(de)
	slog_de = ln(slog_det)

	return sign, slog_de, true 	
}


slog_det_tensor :: proc(	
	$Nd: int, 
	a: md.MdArray($T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	sign: md.MdArray(int, Nd-2),
	slog_de: md.MdArray(T, Nd-2), 
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	slog_de = det(Nd, a, allocator, location) or_return

	sign := md.o_sign(slog_de, allocator, location) or_return

	md.i_abs(slog_de)
	md.i_ln(slog_de)

	return sign, slog_de, true 	
}


slog_det :: proc { det_matrix, det_tensor}


@(private="file")
lapacke_det_matrix :: proc(	
	a: md.MdArray($T, 2),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	de: T, 
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {

	n:= a.shape[0]

	n_b:= lapacke.blasint(n)

	ipiv := make([]lapacke.blasint, n)
	defer delete(ipiv)

	slice.fill(ipiv, 0)
	lapack_lu_wrapper(
		a.buffer,
		n_b, n_b,
		ipiv,
		location
	) or_return

	de = 1

	for i in 0..<n{
		de *= a.buffer[i*n+i]
		if ipiv[i] != lapacke.blasint(i) do de *= -1
	}

	return de, true
}


@(private="file")
lapacke_det :: proc(	
	$Nd: int, 
	a: md.MdArray($T, Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	de: md.MdArray(T, Nd-2), 
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>2 {

	n:= a.shape[Nd-1]

	n_b:= lapacke.blasint(n)

	ipiv := make([]lapacke.blasint, n)
	defer delete(ipiv)

	det_shape : [Nd-2]int

	for d in 0..<Nd-2{
		det_shape[d] = a.shape[d]
	}

	de = md.make_mdarray(T, det_shape, allocator, location) or_return

	a_sig:= n*n

	a_s: []T

	for i in 0..<(md.size(a)/(a_sig)){
		a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
		de.buffer[i] = 1

		slice.fill(ipiv, 0)

		lapack_lu_wrapper(
			a_s,
			n_b, n_b,
			ipiv,
			location
		) or_return

		for j in 0..<n{
			de.buffer[i] *= a_s[j*n+j]
			if ipiv[j] != lapacke.blasint(j) do de.buffer[i] *= -1
		}
	}

	return de, true
}



inv :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 inv_a:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 #optional_ok{

	validate_open_blas(a, allocator, location) or_return

	n:= a.shape[Nd-1]
	if n != a.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"the inverse can only be computed for nonsymteric square matrices or stacks of matrices.",
			location=location,
		)
	}

	a_ := md.copy_array(a, allocator, location) or_return
	
	ok = lapacke_inv(a_, allocator, location)

	if !ok {
		md.free_mdarray(a_)
		logging.error(
			.ArithmeticError,
			"Matrix is singular, it has no inverse",
			location, 
		)
		return 
	}
		
	return a_, true
}


@(private="file")
lapacke_inv :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	n:= a.shape[Nd-1]

	n_b:= lapacke.blasint(n)


	ipiv := make([]lapacke.blasint, n)
	defer delete(ipiv)

	when Nd == 2 {
		lapack_lu_inv_wrapper(
			a.buffer,
			n_b, ipiv,
			location 
		) or_return
	} else { 
		a_sig := n*n
		a_s : []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]

			lapack_lu_inv_wrapper(
				a_s,
				n_b, ipiv,
				location 
			) or_return
		}
	}

	return true
}


pinv :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 pinv_a:md.MdArray(T, Nd),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 #optional_ok{

	s, u, vt := reduced_svd(Nd, a, allocator, location) or_return

	md.i_reciprocal(s)

	s_diag := make_diagonal(s, allocator, location)

	indices : [Nd]int
	when Nd >2 do for i in 0..<Nd-2{
		indices[i] = i
	}
	indices[Nd-2] = Nd-1
	indices[Nd-1] = Nd-2
	
	ut := md.transpose_view(u, indices, location) or_return 
	v := md.transpose_view(vt, indices, location) or_return

	intr := matmul(v, s_diag) or_return
	pinv_a = matmul(intr, ut) or_return
	
	md.free_mdarray(s)
	md.free_mdarray(u)
	md.free_mdarray(vt)
	md.free_mdarray(intr)
	md.free_mdarray(s_diag)
			
	return pinv_a, true
}


solve :: proc(
	a: md.MdArray($T, $Nd),
	b: md.MdArray(T, $Md),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	 solution:md.MdArray(T, Md),
	 ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T),
		Nd>=2 && (Nd==Md || Md==Nd-1) #optional_ok{			

	a_ := md.copy_array(a, allocator, location) or_return		
	defer md.free_mdarray(a_)

	solution = md.copy_array(b, allocator, location) or_return		

	ok = lapacke_solve(a_, solution, allocator, location) 
	if !ok {
		md.free_mdarray(solution)
		return
	}

	return solution, true
}



@(private="file")
lapacke_solve :: proc(	
	a: md.MdArray($T, $Nd),
	b: md.MdArray(T, $Md),
	allocator:= context.allocator,
	location := #caller_location,
) -> (
	ok:bool,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T), Nd>=2 {

	n:= a.shape[Nd-1]
	when Nd == Md {
		nrhs:= b.shape[Md-1]
	} else {
		nrhs := 1
	}

	n_b:= lapacke.blasint(n)
	nrhs_b:= lapacke.blasint(nrhs)

	ipiv := make([]lapacke.blasint, n)
	defer delete(ipiv)

	when Nd == 2 {
		lapack_solve_wrapper(
			a.buffer, b.buffer,
			n_b, nrhs_b, ipiv,
			allocator, location
		) or_return
	} else { 

		a_sig := n*n
		b_sig := nrhs*n

		a_s : []T
		b_s : []T

		for i in 0..<(md.size(a)/(a_sig)){
			a_s = a.buffer[i*a_sig: i*a_sig+a_sig]
			b_s = b.buffer[i*b_sig: i*b_sig+b_sig]

			lapack_solve_wrapper(
				a_s, b_s,
				n_b, nrhs_b, ipiv,
				allocator, location
			) or_return
		}
	}

	return true
}


