package main

import "slow_works"
import "simd_version"

Version :: enum {
	SIMD,
	SLOW,
}

main :: proc() {
	version:Version = .SLOW
	switch version {
	case .SIMD:
		simd_version.run()
	case .SLOW:
		slow_works.run()
	}
}
