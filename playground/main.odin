package main

import "core:terminal"
import "core:fmt" 
import "core:simd"
import "core:math/rand"
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
	rand.reset(1)
	input_matrix := make_slice([]simd_4, ITERATIONS)
	defer {
		delete(input_matrix)
	}
	index := 0
	for i in 0..<ITERATIONS {
		val := rand.float64_range(0.,MAX_INPUT_VALUE)
		input_matrix[i] = {
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE)
		}
		index += 1
	}

	expected_vector := make_slice([]int, ITERATIONS)
	defer delete(expected_vector)
	index = 0
	for &entry, i in expected_vector {
		index, max := find_min(simd.to_array(input_matrix[i]))
		entry = index
	}
	hidden_layer1: Hidden_Layer
	hidden_layer1.w1 = random_matrix4x4()
	hidden_layer1.w2 = random_matrix4x4()
	hidden_layer1.w3 = random_matrix4x4()
	hidden_layer1.w4 = random_matrix4x4()

	out_layer: Output_Layer
	out_layer.w1 = random_matrix4x1()

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	for input in input_matrix {

		hidden_layer1 := hidden_layer1
		out_layer := out_layer  
		v := transmute(matrix[1,4]f64)input

		hidden_layer1.neurons = transmute(simd_16) [4]matrix[1,4]f64{
			v * hidden_layer1.w1,
			v * hidden_layer1.w2,
			v * hidden_layer1.w3,
			v * hidden_layer1.w4,
		}
		hidden_layer1.neurons = relu_16(hidden_layer1.neurons) 
		hidden_layer1.neurons = normalize_simd(hidden_layer1.neurons)

		o_v := transmute(matrix[4, 4]f64)hidden_layer1.neurons
		out_layer.neurons = transmute(simd_4)(o_v * out_layer.w1)
		out_layer.neurons = normalize_simd(out_layer.neurons)
		//TODO: soft max
		//TODO: Back prop
	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	fmt.println("DONE")
}

forward_prop :: proc(input: simd_4, hidden_layer1: Hidden_Layer, out_layer: Output_Layer) {
}

dot :: proc(v: simd_16, m: []simd_16) -> [8]f64 {
	temp_m: [8]simd_16
	output: [8]f64

	for row, i in m {
		temp_m[i] = simd.mul(v, row)
		output[i] = simd.reduce_add_bisect(temp_m[i])
	}

	return output
}

relu_16 :: proc(v: simd_16) -> simd_16 {
	zeros := simd_16{}
	return simd.max(zeros, v)
}

find_min :: proc(vector: [4]f64) -> (int, f64) {
	curr_i := -1
	curr_min := 99999999.
	for n, i in vector {
		if n < curr_min {
			curr_min = n
			curr_i = i
		}
	}
	return curr_i, curr_min
}

random_matrix4x4 :: proc() -> matrix[4,4]f64 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

random_matrix4x1 :: proc() -> matrix[4,1]f64 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

normalize_simd16 :: proc(v: simd_16) -> simd_16 {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, 16)
	return _v
}

normalize_simd4 :: proc(v: simd_4) -> simd_4 {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, 4)
	return _v
}

normalize_simd :: proc(v: $T) -> T {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, len(_v))
	return _v
}

softmax_simd4 :: proc(v: simd_4) -> simd_4 {
	_v := v
	a := simd.to_array(_v)
	/* simd.reduce_add_bisect */
	return _v
}

normalize_vector :: proc(vector: []f64) -> []f64 {
	vector := vector
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
	return vector
}
