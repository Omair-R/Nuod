# Nuod
Nuod (Numerical Odin) is an [Odin](https://odin-lang.org/) library for creating and manipulating numerical multi-dimensional arrays. 
It is heavily inspired by the [Numpy](https://github.com/numpy/numpy) python library.

> [!CAUTION]
> Noud is a work in progress.

## Features
- A flexible multi-dimensional array type with various manipulation procedures.
- Various mathimatical procedures. 
- Basic linear algebra procedures.

## Simple Example

```odin
package main

import "core:log"
import "core:fmt"
import md "nuod/mdarray"
import ml "nuod/linalg"
import rn "nuod/random"
import "core:math/rand"

main :: proc () {
  // create a console logger to view log messages created by the library.
	logger:= log.create_console_logger()
	context.logger = logger
	defer log.destroy_console_logger(logger)

  // you can make use of the provided odin-compatible random number generators.
	context.random_generator = rn.pcg_random_generator()
	rand.reset(64)

  // create a random array of type f64 with a dimensions (2, 3, 3)
	arr := rn.random_float(f64, shape=[3]int{2, 3, 3})

  // get a view slice of the first matrix of dimensions (3, 3)
	first_matrix := md.slice_view(3, arr, index=0)

  // transpose the first matrix
	trans_matrix := md.transpose_copy(first_matrix)

  // perform matrix multiplication on the two matrices.
	inner_product := ml.matmul(trans_matrix, first_matrix)

	fmt.println("Inner product: ")
	md.println(inner_product)

  // free the created arrays.
	md.free_mdarray(arr)
	md.free_mdarray(trans_matrix)
	md.free_mdarray(inner_product)
}

```
## Acknowledgment
Special thanks to [Kalsprite](https://github.com/kalsprite) for providing [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) [bindings](https://github.com/kalsprite/odin-openblas).
## Todo List
- [ ] create user-defined formats for MdArray and register it.
- [ ] comment documentation.
- [ ] write the rest of the tests.
- [ ] add matrix decomposition operations (e.g. QR, SVD...).
- [ ] and more!
