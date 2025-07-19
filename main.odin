package main

import "slow_works"
import "simd_version"
import "matrix_version"

Version :: enum {
	SIMD,
	SLOW,
	MATRIX,
}

main :: proc() {
	version:Version = .MATRIX
	switch version {
	case .SIMD:
		simd_version.run()
	case .SLOW:
		slow_works.run()
	case .MATRIX:
		matrix_version.run()
	}
}
