package linalg

import md "../mdarray"
import "../logging"
import "../lapacke"


@private
validate_openblas_supported :: proc(	
	location := #caller_location,
) -> (ok: bool) {
	if !lapacke.OPENBLAS_SUPPORTED {
		logging.error(
			.NotImplemented,
			"Used linear algebra operation is only implemented with openblas support.",
			location,
		)
		return
	}

	return true
}

@private
validate_square :: proc(	
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (ok: bool) where Nd>=2 {
	if a.shape[Nd-1] != a.shape[Nd-2] {
		logging.error(
			.ArguementError,
			"the used operator accepts only square matrices or stacks of matrices.",
			location=location,
		)
	}

	return true
}


@private
validate_general_openblas :: proc(
	a: md.MdArray($T, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (ok: bool) {
	
	md.validate_initialized(a, location) or_return

	validate_openblas_supported(location) or_return

	when !(T == f32 || T == f64 || T == complex64 || T== complex128){
		logging.error(
			.ArguementError,
			"openblas routines do not support half precision routines.",
			location,
		)
		return
	}

	return true
}


@private
validate_eig_operators :: proc(
	$C : typeid,
	a: md.MdArray($F, $Nd),
	allocator:= context.allocator,
	location := #caller_location,
) -> (ok: bool) {
	
	md.validate_initialized(a, location) or_return

	validate_openblas_supported(location) or_return

	when !(F == f32 || F == f64){
		logging.error(
			.ArguementError,
			"openblas routines do not support half precision routines.",
			location,
		)
		return
	}

	when (F == f32 && C != complex64) || (F == f64 && C != complex128){
		logging.error(
			.ArguementError,
			"the complex type must correspond to the float.",
			location,
		)
		return
	}
	
	validate_square(a, allocator, location) or_return

	return true
}


@private
validate_inner_dimensions :: proc(
	a: md.MdArray($T, $Nd),
	b: md.MdArray(T, $Md),
	allocator:= context.allocator,
	location := #caller_location,
) -> (ok: bool) {
	
	n := a.shape[Nd-2]
	when Nd == Md {
		k := b.shape[Md-2]
	} else {
		k := b.shape[Md-1]
	}

	if n != k {
		logging.error(
			.ArguementError,
			"Input arrays passed with inconsistent dimensions.",
			location
		)

		return
	}

	return true
}

