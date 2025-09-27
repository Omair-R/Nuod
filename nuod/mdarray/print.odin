package mdarray

import "core:os"
import "core:fmt"
import "core:slice"

_fprint_2d ::proc(
	handle: os.Handle,
	mdarray :MdArray($T, $Nd),
	last_dim: int,
	indent:int,
	newline:bool=false,
	flush:bool=true
) {
	n_sig : int = size(mdarray)/last_dim

	fmt.fprint(handle, "[", flush=false)

	for i in 0..<n_sig {
		if i != 0 {
			for _ in 0..<indent {
				fmt.fprint(handle, " ", flush=false)
			}
		}

		fmt.fprint(handle, "[", flush=false)
		i_end := i*last_dim + last_dim
		for i_dim:= i*last_dim; i_dim < i_end; i_dim += 1{
			fmt.fprint(handle, get_linear(mdarray, i_dim), flush=false)
			if i_dim != i_end-1{
				fmt.fprint(handle, ", ", flush = false)
			}
		}
		fmt.fprint(handle, "]", flush=false)

		if i != n_sig-1{
			fmt.fprintln(handle, ",", flush = false)
		}
	}

	if newline{
		fmt.fprintln(handle, "]", flush=true)
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
	
	when Nd ==1 {				
		if newline{
			fmt.fprintln(handle, mdarray.buffer)
		}else {
			fmt.fprint(handle, mdarray.buffer)
		}		
	} else when Nd==2{
		offset := 0
		last_dim := mdarray.shape[Nd-1]
		_fprint_2d(handle, mdarray, last_dim, indent=indent, newline=newline)
	} else {				
		first_dim := mdarray.shape[0]

		fmt.fprint(handle, "[", flush=false)

		for i in 0..<first_dim{

			sliced_view, ok := slice_view(Nd, mdarray, i)
			if !ok { return }

			_fprint_inner(handle, sliced_view, indent, false)

			if i != first_dim - 1 {
				fmt.fprintln(handle, ",", flush = false)
				fmt.fprintln(handle, "", flush = false)

				indent := indent-ndim(mdarray)+2
				for _ in 0..<indent{
					fmt.fprint(handle, " ", flush = false)
				}
			}
		}


		if newline {
			fmt.fprintln(handle, "]", flush=true)
			return
		}

		fmt.fprint(handle, "]", flush=true)
	}
	return
}


fprint :: proc(handle: os.Handle, mdarray: MdArray($T, $Nd)){
	if is_none(mdarray) {return } 
	indent := ndim(mdarray)-1
	_fprint_inner(handle, mdarray, indent)
}


fprintln :: proc(handle: os.Handle, mdarray: MdArray($T, $Nd)){
	if is_none(mdarray) {return } 
	indent := ndim(mdarray) -1
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
