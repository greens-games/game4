package main

import "core:terminal"
import "core:fmt" 
import "core:simd"
import "core:math/rand"
import "core:math/linalg"
import "core:math"
import "base:intrinsics"
import "core:time"

simd_16 :: #simd[16]f64
simd_8 :: #simd[8]f64
simd_4 :: #simd[4]f64

Input_Layer :: struct {

}


Hidden_Layer :: struct {
	w1: matrix[4, 4]f64,
	w2: matrix[4, 4]f64,
	w3: matrix[4, 4]f64,
	w4: matrix[4, 4]f64,
	neurons: #simd[16]f64,
}

Output_Layer :: struct {
	w1: matrix[4, 1]f64,
	neurons: #simd[4]f64,
}

ITERATIONS :: 1
MAX_INPUT_VALUE :: 100.
MAX_WEIGHT_VALUE :: 5.

main :: proc() {

	a := matrix[4,4]f64 {
	2,2,2,2,
	2,2,2,2,
	2,2,2,2,
	2,2,2,2,
	}

	b := matrix[4,4]f64 {
	3,3,3,3,
	3,3,3,3,
	3,3,3,3,
	3,3,3,3,
	}

	fmt.println( b * a)
	/* fmt.println(linalg.matrix_mul(a,b)) */
	fmt.println(linalg.matrix_comp_mul(a, b))

}

normalize_vector :: proc(v: $T) -> T {
	vector := simd.to_array(v)
	fmt.println("V: ", v)
	min := 99999.
	max := -99999.
	for val in vector {
		if val < min {
			min = val
		}

		if val > max {
			max = val
		}
	}

	denom := max - min
	for &val in vector {
		numerator := val - min
		range := 1. - -1.
		val = (numerator/denom) * range + 1
	}
	fmt.println("VECTOR: ", vector)
	return simd.from_array(vector)
}

generate_simd4 :: proc(val: $T) -> #simd[4]T{
	return {
	val, val, val, val,
	}
}
