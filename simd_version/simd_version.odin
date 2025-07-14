package simd_version

import "core:fmt" 
import "core:simd"
import "core:math/rand"
import "core:math"
import "base:intrinsics"
import "core:time"
import "core:mem"

simd_16 :: #simd[16]f64
simd_8 :: #simd[8]f64
simd_4 :: #simd[4]f64

Classification :: enum {
	WORK,
	SLEEP,
	GATHER,
	TRAIN,
	/* FREE, */
}

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
	w1: [4]simd_16,
	neurons: #simd[4]f64,
}

ITERATIONS :: 1000000
MAX_INPUT_VALUE :: 100.
MAX_WEIGHT_VALUE :: 5.

run :: proc() {
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)
		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not free: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}
	rand.reset(1)
	input_matrix := make_slice([]simd_4, ITERATIONS)
	defer {
		delete(input_matrix)
	}

	expected_vector := make_slice([]Classification, ITERATIONS)
	defer delete(expected_vector)

	index := 0
	for i in 0..<ITERATIONS {
		input_matrix[i] = {
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE)
		}
		index, max := find_min(simd.to_array(input_matrix[i]))
		expected_vector[i] = Classification(index)
		index += 1
	}

	index = 0

	hidden_layer1: Hidden_Layer
	hidden_layer1.w1 = random_matrix4x4()
	hidden_layer1.w2 = random_matrix4x4()
	hidden_layer1.w3 = random_matrix4x4()
	hidden_layer1.w4 = random_matrix4x4()

	out_layer: Output_Layer
	for &weights in out_layer.w1 {
		weights = random_simd16()
	}
	alpha := 0.1

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	for input, index in input_matrix {

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

		//TODO: This will need to be cleaned up a bit but should in theory be ok
		out_layer_temp: [4]f64
		for weights, i in out_layer.w1 {
			out_layer_temp[i] = dot_simd16(weights, hidden_layer1.neurons)
		}
		out_layer.neurons = {out_layer_temp[0], out_layer_temp[1], out_layer_temp[2], out_layer_temp[3]}
		out_layer.neurons = normalize_simd(out_layer.neurons)

		//Soft max
		a := simd.to_array(out_layer.neurons)
		ret: [len(Classification)]f64
		exp_sum:f64 = 0.
		for val in a {
			exp_sum += math.exp(val)
		}
		for val, index in a {
			ret[index] = math.exp(val)/exp_sum
		}
		
		//TODO: Back prop
		
		//Cross entropy loss
		expected := ret[int(expected_vector[index])]
		loss := -(math.log10(expected))
		update_val := generate_simd16(alpha * loss)
		
		for &weights in out_layer.w1 {
			weights = simd.sub(weights, update_val)
		}
		

		vals:simd_16 = auto_cast d_relu_simd16(hidden_layer1.neurons)
		comb_loss := alpha * loss
		vals = simd.mul(vals, generate_simd16(comb_loss))

		staging1 := transmute(simd_16)hidden_layer1.w1
		staging1 = simd.sub(staging1, vals)
		hidden_layer1.w1 = transmute(matrix[4,4]f64)staging1

		staging2 := transmute(simd_16)hidden_layer1.w2
		staging2 = simd.sub(staging2, vals)
		hidden_layer1.w2 = transmute(matrix[4,4]f64)staging2

		staging3 := transmute(simd_16)hidden_layer1.w3
		staging3 = simd.sub(staging3, vals)
		hidden_layer1.w3 = transmute(matrix[4,4]f64)staging3

		staging4 := transmute(simd_16)hidden_layer1.w4
		staging4 = simd.sub(staging4, vals)
		hidden_layer1.w4 = transmute(matrix[4,4]f64)staging4

	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	fmt.println("DONE")
}

forward_prop :: proc(input: simd_4, hidden_layer1: Hidden_Layer, out_layer: Output_Layer) {
}

dot_simd16 :: proc(v1: simd_16, v2: simd_16) -> f64 {
	return simd.reduce_add_bisect(simd.mul(v1, v2))
}

relu_16 :: proc(v: simd_16) -> simd_16 {
	zeros := simd_16{}
	return simd.max(zeros, v)
}

d_relu_simd16 :: proc(v: simd_16) -> #simd [16]u64 {
	_v:#simd [16]u64 = auto_cast simd.floor(v)
	return simd.lanes_ge(_v , #simd [16]u64{})
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

random_simd16 :: proc() -> simd_16 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

generate_simd16 :: proc(val: f64) -> simd_16 {
	return {
	val, val, val, val,
	val, val, val, val,
	val, val, val, val,
	val, val, val, val,
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

//NOTE: This probably will need to change to match slow_version
normalize_simd :: proc(v: $T) -> T {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, len(_v))
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
