package main

import "slow_works"
import "simd_version"

main :: proc() {
	simd_version.run()
	slow_works.run()
}
