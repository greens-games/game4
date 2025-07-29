package matrix_version

import "core:fmt" 
import "core:simd"
import "core:math/rand"
import "core:math/linalg"
import "core:math"
import "base:intrinsics"
import "core:time"
import "core:mem"
import "core:os"
import "core:strconv"

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

H_Layer :: struct {
	weights: matrix[4,4]f64,
	biases: matrix[1,4]f64,
	weighted_sums: matrix[1,4]f64,
	neurons: matrix[1,4]f64,
}

O_Layer_Classification :: struct {
	weights: matrix[4,4]f64,
	biases: matrix[1,4]f64,
	weighted_sums: matrix[1,4]f64,
	neurons: matrix[1,4]f64,
}

ITERATIONS :: 10
MAX_INPUT_VALUE :: 100.
MAX_WEIGHT_VALUE :: 1.
MIN_WEIGHT_VALUE :: -1.
NUM_LAYERS :: 1
ALPHA :: 0.1

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

	h_layer1: H_Layer
	h_layer1.weights = random_matrix4x4()
	fmt.println("H_LAYER1: ", h_layer1, "\n")
	/* h_layer1.biases = random_matrix1x4() */
	o_layer: O_Layer_Classification
	o_layer.weights = random_matrix4x4()
	fmt.println("O_LAYER: ", o_layer, "\n")
	/* o_layer.biases = random_matrix1x4() */

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	ret:[len(Classification)]f64
	for input, index in input_matrix {
		/* fmt.println("ITERATION: ", index) */

		ret = forward_prop(input, &h_layer1, &o_layer)
		fmt.println(ret)
		/* i, max := find_max(ret)
		fmt.println("RET: ", Classification(i)) */
		
		//TODO: Back prop
		/* back_prop(input, ret, &h_layer1, &o_layer, expected_vector, index) */
		//Cross entropy loss

		fmt.println()
	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	fmt.println("DONE TRAINING")

	/* running := true
	input_buff: [1024]byte
	data: [4]f64
	for running {
		
		fmt.print("WEALTH:")
		os.read(os.stdin, input_buff[:])
		data[0] = strconv.atof(string(input_buff[:3]))
		fmt.print("HEALTH:")
		os.read(os.stdin, input_buff[:])
		data[1] = strconv.atof(string(input_buff[:3]))

		fmt.print("SUPPLIES:")
		os.read(os.stdin, input_buff[:])
		data[2] = strconv.atof(string(input_buff[:3]))

		fmt.print("STENGTH:")
		os.read(os.stdin, input_buff[:])
		data[3] = strconv.atof(string(input_buff[:3]))

		simd_data := simd.from_array(data) 
		o := forward_prop(simd_data, &hidden_layer1, &out_layer)
		i, max := find_max(o)
		fmt.println(Classification(i))
		//TODO: Add some runtime training this means we need to figure out the expted value for a given state
	}  */
}

foward_hidden_layer :: proc(input: simd_4, h_layer: ^H_Layer) {
	v := transmute(matrix[1,4]f64)normalize_vector(input)

	h_layer.weighted_sums = v * h_layer.weights
	/* h_layer.weighted_sums = h_layer.weighted_sums + h_layer.biases */
	h_layer.weighted_sums = transmute(matrix[1,4]f64)normalize_vector(transmute(simd_4)h_layer.weighted_sums)
	h_layer.neurons = transmute(matrix[1,4]f64)relu(transmute(simd_4)h_layer.weighted_sums)
	h_layer.neurons = transmute(matrix[1,4]f64)normalize_vector(transmute(simd_4)h_layer.neurons)
}

forward_output :: proc(prev_layer: H_Layer, out_layer: ^O_Layer_Classification) {

	out_layer.weighted_sums = prev_layer.neurons * out_layer.weights
	/* out_layer.weighted_sums = h_layer1.weighted_sums + out_layer.biases */
	out_layer.weighted_sums = transmute(matrix[1,4]f64)normalize_vector(transmute(simd_4)out_layer.weighted_sums)
	out_layer.neurons = transmute(matrix[1,4]f64)soft_max(transmute([len(Classification)]f64)out_layer.weighted_sums)
	/* o_layer.neurons = transmute(matrix[1,4]f64)normalize_vector(transmute(simd_4)o_layer.neurons) */
}

forward_prop :: proc(input: simd_4, h_layer1:^H_Layer, o_layer: ^O_Layer_Classification) -> [len(Classification)]f64 {
	_input := input
	_input = normalize_vector(_input)

	foward_hidden_layer(input, h_layer1)
	forward_output(h_layer1^, o_layer)

	//Soft max
	ret := transmute([len(Classification)]f64)o_layer.neurons
	return ret
}

back_prop :: proc(input: simd_4, ret: [len(Classification)]f64, h_layer1: ^H_Layer, o_layer: ^O_Layer_Classification, expected_vector: []Classification, index: int) {
	/* temp_out_loss_v := simd.mul(out_loss_v, hidden_layers[0].neurons) */
	one_hot := [len(Classification)]f64{}
	//NOTE: Might still be able to use log(ret[int(expected_vector[index](]) instead of just 1 and everything else is 0
	one_hot[int(expected_vector[index])] = -math.log10(ret[int(expected_vector[index])])
	dz2:matrix[1,4]f64 = transmute(matrix[1,4]f64)(ret - one_hot)

	//CHANGE IN OUTPUT LAYER
	temp_dz2 := transmute(f64)(dz2 * linalg.transpose(h_layer1.neurons))
	dw2 := 1/f64(ITERATIONS) * temp_dz2
	/* db2 := 1/(ITERATIONS * simd.reduce_add_bisect(transmute(simd_4)dz2)) */

	
	//CHANGE IN HIDDEN LAYER
	d_relu_vec:simd_4 = d_relu(transmute(simd_4)h_layer1.weighted_sums)
	dz1 := linalg.matrix_comp_mul(linalg.transpose(o_layer.weights) * linalg.transpose(dz2), transmute(matrix[4,1]f64)d_relu_vec)
	_input := transmute(matrix[1,4]f64)input
	thing := linalg.transpose(dz1) * linalg.transpose(_input)
	dw1 := 1/f64(ITERATIONS) * transmute(f64)(linalg.transpose(dz1) * linalg.transpose(_input))
	/* db1 := 1/(ITERATIONS * simd.reduce_add_bisect(transmute(simd_4)dz1)) */

	h_layer1.weights = h_layer1.weights - generate_matrix4x4(ALPHA * dw1)
	/* h_layer1.biases = h_layer1.biases - generate_matrix1x4(ALPHA * db1) */
	
	o_layer.weights = o_layer.weights - generate_matrix4x4(ALPHA * dw2)
	/* o_layer.biases = o_layer.biases - generate_matrix1x4(ALPHA * db2) */
}

soft_max :: proc(neurons: [len(Classification)]f64) -> [len(Classification)]f64 {
		ret: [len(Classification)]f64
		exp_sum:f64 = 0.
		for val in neurons {
			exp_sum += math.exp(val)
		}
		for val, index in neurons {
			ret[index] = math.exp(val)/exp_sum
		}
	return ret
}

relu :: proc(v: $T) -> T {
	zeros := T{}
	return simd.max(zeros, v)
}

d_relu :: proc(z: simd_4) -> simd_4 {
	arr := simd.to_array(z)
	for &val in arr {
		val = val >= 0. ? 1. : 0.
	}
	return simd.from_array(arr)
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

find_max :: proc(vector: [4]f64) -> (int, f64) {
	curr_i := -1
	curr_max := -99999999.
	for n, i in vector {
		if n > curr_max {
			curr_max = n
			curr_i = i
		}
	}
	return curr_i, curr_max
}

random_matrix4x4 :: proc() -> matrix[4,4]f64 {
	return {
	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),
	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),
	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),
	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),
	}
}

generate_matrix4x4 :: proc(val: f64) -> matrix[4,4]f64 {
	return {
	val,	val,	val,	val,
	val,	val,	val,	val,
	val,	val,	val,	val,
	val,	val,	val,	val,
	}
}

generate_matrix1x4 :: proc(val: f64) -> matrix[1,4]f64 {
	return {
	val,	val,	val,	val,
	}
}


random_matrix4x1 :: proc() -> matrix[4,1]f64 {
	return {
	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),
	}
}

random_matrix1x4 :: proc() -> matrix[1,4]f64 {
	return {
	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),	rand.float64_range(MIN_WEIGHT_VALUE,MAX_WEIGHT_VALUE),
	}
}

normalize_vector :: proc(v: $T) -> T {
	vector := simd.to_array(v)
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
		val = (numerator/denom) * range - 1
	}
	return simd.from_array(vector)
}

/*
TEST INPUTS
INPUT_MATRIX:  [[40.280407544550926, 26.978681299830352, 94.89336279946937, 41.330696086138097]]

EXPECTED_VECTOR:  [1]

LAYER1:  Layer{weights = [[2.9786259555492247, 1.8490057036625396, 2.54573101301464, 2.008264812816276], [4.603908068634691, 3.8170316246917197, 3.6184763617625215, 3.248871346980866], [4.8353490128955858, 1.140294029915752, 4.5653781245126126, 4.5062585871972356], [1.0528989834503533, 1.5893436276420325, 1.2401304977282601, 4.8967093126096666]]}

OUTPUT_LAYER:  Layer{weights = [[2.236615639505719, 3.3104835621730437, 3.1796061350846254, 1.0705070004598154], [3.0881945081994795, 2.8629947570767493, 1.5491868130513775, 0.40377340971426423], [2.7150350823951297, 1.7340252892460029, 2.4675006487994238, 0.37353848144743407], [2.1309494355081893, 4.946177092888888, 2.386578267238466, 4.7480799156892326]]}
*/

/*
INPUT_MATRIX:  [<40.280407544550926, 26.978681299830352, 94.89336279946937, 41.330696086138097>]

EXPECTED_VECTOR:  [1]

H_LAYER1:  H_Layer{weights = matrix[2.9786259555492247, 1.8490057036625396, 2.54573101301464, 2.008264812816276; 4.603908068634691, 3.8170316246917197, 3.6184763617625215, 3.248871346980866; 4.8353490128955858, 1.140294029915752, 4.5653781245126126, 4.5062585871972356; 1.0528989834503533, 1.5893436276420325, 1.2401304977282601, 4.8967093126096666], biases = matrix[0, 0, 0, 0], weighted_sums = matrix[0, 0, 0, 0], neurons = matrix[0, 0, 0, 0]}

O_LAYER:  O_Layer_Classification{weights = matrix[2.236615639505719, 3.3104835621730437, 3.1796061350846254, 1.0705070004598154; 3.0881945081994795, 2.8629947570767493, 1.5491868130513775, 0.40377340971426423; 2.7150350823951297, 1.7340252892460029, 2.4675006487994238, 0.37353848144743407; 2.1309494355081893, 4.946177092888888, 2.386578267238466, 4.7480799156892326], biases = matrix[0, 0, 0, 0], weighted_sums = matrix[0, 0, 0, 0], neurons = matrix[0, 0, 0, 0]}
*/
