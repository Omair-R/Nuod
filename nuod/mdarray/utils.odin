package mdarray

import "../errors"
import "../logging"

swap :: #force_inline proc (a : ^$T, b: ^$S){
	temp: T = a^
	a^ = b^
	b^ = temp
}


compute_strides :: proc(shape: [$Nd]int) -> (strides:[Nd]int){	
	for s in 0..<len(strides) {
		strides[s] = 1
		for t in s+1..<len(strides){
			strides[s] *= shape[t]
		}
	}

	return strides
}


validate_initialized :: #force_inline proc(mdarray: MdArray($T, $Nd), location:=#caller_location) -> bool {	
	if is_none(mdarray) {
		logging.warning(errors.NuodWarning.UninitializedArrayWarning, location = location)
		return false
	}
	return true
}


validate_positive_shape :: proc(shape: [$Nd]int, location:=#caller_location) -> bool {	
	for dim in shape {
		if dim <= 0 {
			logging.error(.ArguementError, "Attempted to make an array with shape < 0.", location=location)
			return  false
		}
	}
	return true
}


validate_shape_and_get_size :: proc(shape: [$Nd]int, location:=#caller_location) -> (size:int, ok:bool){	
	size = 1
	for dim in shape {
		if dim <= 0 {
			logging.error(.ArguementError, "Attempted to make an array with shape < 0.", location=location)
			return 0, false
		}
		size *= dim
	}
	return size, true
}


validate_pos_within_shape :: proc(pos: [$Nd]int, shape: [Nd]int, location:=#caller_location)-> bool {
	for i in 0..<Nd{
		if pos[i] < 0 || pos[i] >= shape[i]{
			logging.error(.ArguementError, "Provided axis cannot be bigger than the number of dimensions.", location=location)
			return false
		}  
	}

	return true
}


validate_shape_match :: proc(a: MdArray($T, $Nd), b: MdArray($S, Nd), location:=#caller_location) -> bool {
	for d in 0..<Nd{
		if a.shape[d] != b.shape[d]{
			logging.error(.ArguementError, "Mismatched dimensions are not allowed.", location=location)
			return false
		}
	}

	return true
}
