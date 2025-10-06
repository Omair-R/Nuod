package cblas

import "core:c"


when ODIN_OS == .Windows || ODIN_OS == .Darwin || ODIN_OS == .Linux{
	OPENBLAS_SUPPORTED :: true
	VERSION :: "OpenBLAS 0.3.30"
	
} else {
	OPENBLAS_SUPPORTED :: false
}

bfloat16 :: u16
blasint :: i64

