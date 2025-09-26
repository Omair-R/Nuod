package mdarray

import "core:os"
import "core:fmt"
import "core:slice"

_fprint_2d ::proc(
	handle: os.Handle,
	buffer :[]$T,
	last_dim: int,
	indent:int,
	newline:bool=false,
	flush:bool=true
) {
	n_sig : int = len(buffer)/last_dim

	fmt.fprint(handle, "[", flush=false)

	for i in 0..<n_sig {
		if i != 0 {
			for _ in 0..<indent {
				fmt.fprint(handle, " ", flush=false)
			}
		}
		i_dim := i * last_dim

		fmt.fprint(handle, buffer[i_dim:i_dim+last_dim], flush=false)

		if i != n_sig-1{
			fmt.fprintln(handle, ",", flush = false)
		}
	}

	if newline {
		fmt.fprintln(handle, "]", flush=flush)
		return
	}

	fmt.fprint(handle, "]", flush=flush)
}


_fprint_inner :: proc(
	handle: os.Handle,
	mdarray :MdArray($T, $Nd),
	indent : int,
	newline: bool =false
) {
	switch ndim(mdarray) {
	case 1:
		if newline{
			fmt.fprintln(handle, mdarray.buffer)
		}else {
			fmt.fprint(handle, mdarray.buffer)
		}		
	case 2:
		last_dim := mdarray.shape[ndim(mdarray)-1]
		_fprint_2d(handle, mdarray.buffer, last_dim, 1, newline=newline)
	case :
		//TODO
	}

	return
}


fprint :: proc(handle: os.Handle, mdarray: MdArray($T, $Nd)){
	if is_none(mdarray) {return } 
	indent := ndim(mdarray) -2
	_fprint_inner(handle, mdarray, indent)
}


fprintln :: proc(handle: os.Handle, mdarray: MdArray($T, $Nd)){
	if is_none(mdarray) {return } 
	indent := ndim(mdarray) -2
	_fprint_inner(handle, mdarray, indent, newline=true)
}


print :: proc (mdarray: MdArray($T, $Nd)){
	fprint(os.stdout, mdarray)
}


println :: proc (mdarray: MdArray($T, $Nd)){
	fprintln(os.stdout, mdarray)
}


eprint :: proc (mdarray: MdArray($T, $Nd)){
	fprint(os.stderr, mdarray)
}


eprintln :: proc (mdarray: MdArray($T, $Nd)){
	fprintln(os.stderr, mdarray)
}
