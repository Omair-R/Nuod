<img width="1391" height="376" alt="Nuod_cover" src="https://github.com/user-attachments/assets/64d3606f-7d58-40d7-a448-37789cc4fe06" />

-----
Nuod (Numerical Odin) is an [Odin](https://odin-lang.org/) package for creating, manipulating, and performing numerical operations on multi-dimensional arrays. 
It is inspired by the [Numpy](https://github.com/numpy/numpy) Python library.

> [!IMPORTANT]
> Noud is now in early access. It does pass all the tests; however, the tests are still limited. If you face any bugs, unexpected errors, or issue, please report it in the Issues section.

## Features
- A flexible multi-dimensional array type with various manipulation procedures.
- Various mathematical procedures.
- Multi-dimensional fast Fourier transform procedures.
- Basic linear algebra procedures.
- Several matrix factorization techniques.
- Random number methods with various generators.

## [Documentation](https://github.com/Omair-R/pkg_odin)
The package is mostly documented through comments in the source code. However, by utilizing a fork of the official Odin documentation tool, Nuod has a static [documentation site](https://github.com/Omair-R/pkg_odin) set up to help with navigation.

## Usage
To add Nuod to your project, you may simply make a copy of the "nuod" folder and add it to your project's repository. Aside from the libraries and packages provided by the Odin language, Nuod makes use of only one dependency: OpenBLAS. For MacOS and Linux-based systems, installing the OpenBLAS library may be done globally for ease of use. However, in the case of Windows, copies of the static and dynamic OpenBLAS library files must be included. To facilitate usage, Nuod along with the required OpenBlas files are provided in the release section of this repository. 

To install the OpenBlas dependency you may follow:

### Linux

#### Debian-based
```
  sudo apt update
  sudo apt install libopenblas-dev
```

#### Fedora-based 
```
  sudo dnf check-update
  sudo dnf install openblas openblas-devel
```

#### Arch-based
```
  sudo pacman -S openblas
```

#### OpenSUSE-based
```
  sudo zypper refresh
  sudo zypper install openblas-devel
```

### MacOS

#### Homebrew
```
  brew install openblas
```

#### MacPorts
```
  sudo port install OpenBLAS-devel
```

### FreeBSD
```
  pkg install openblas
```
### Windows
To use Nuod with Windows, you may use the "Nuod_win" release archive in the Releases section. The current release tag is v0.0.1.


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

***more examples will soon be provided!***

## Running Tests
Odin makes testing quite an easy task. Since Nuod has several subpackages you can run the tests by running the following command in your terminal:
```
  odin test tests/ -all-packages
```

## Acknowledgment
Special thanks to [Kalsprite](https://github.com/kalsprite) for providing [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) [bindings](https://github.com/kalsprite/odin-openblas).

