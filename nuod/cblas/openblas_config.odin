package cblas

import "core:c"


when ODIN_OS == .Windows {
	foreign import openblas "../../vendors/openblas/lib/libopenblas.lib"
} else when ODIN_OS == .Linux {
	foreign import lib "system:openblas"
}

// CONFIG_H ::
OS_WINNT :: 1
ARCH_ARM64 :: 1
C_Clang :: 1
_64BIT__ :: 1
// FUNDERSCORE ::
BUNDERSCORE :: "_"
NEEDBUNDERSCORE :: 1
NEED2UNDERSCORES :: 0
// ARMV8 ::
// HAVE_NEON ::
// HAVE_VFPV4 ::
// NEOVERSEN1 ::
L1_CODE_SIZE :: 65536
L1_CODE_LINESIZE :: 64
L1_CODE_ASSOCIATIVE :: 4
L1_DATA_SIZE :: 65536
L1_DATA_LINESIZE :: 64
L1_DATA_ASSOCIATIVE :: 4
L2_SIZE :: 1048576
L2_LINESIZE :: 64
L2_ASSOCIATIVE :: 8
DTB_DEFAULT_ENTRIES :: 48
DTB_SIZE :: 4096
CHAR_CORENAME :: "NEOVERSEN1"
// USE64BITINT ::
GEMM_MULTITHREAD_THRESHOLD :: 4
VERSION :: "OpenBLAS 0.3.30"

// WINDOWS_ABI ::
// OS_WINDOWS ::

xdouble :: f64

BLASLONG :: i64

BLASULONG :: u64

bfloat16 :: u16

blasint :: i64

FLOATRET :: f32

// COMPLEX_C99 ::

openblas_complex_float :: complex64

openblas_complex_double :: complex128

openblas_complex_xdouble :: complex128
