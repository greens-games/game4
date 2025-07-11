package slow_works

import "core:fmt"
import "core:os"
import "core:strings"
import "core:strconv"
import "core:mem"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:time"

Classification :: enum {
	WORK,
	SLEEP,
	GATHER,
	TRAIN,
	/* FREE, */
}

State :: struct {
	wealth: f64,
	health: f64,
	supplies: f64,
	strength: f64,
}

Layer :: struct {
	weights: Matrix
}

Matrix :: distinct [][]f64

//MODEL PARAMS
ITERATIONS :: 100000
NUM_LAYERS :: 2

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
	//Init layers
	layer1: Layer
	layer1.weights = create_matrix(5,4)
	defer {
		clean_matrix(layer1.weights)
	}
	for r in 0..<len(layer1.weights) {
		for c in 0..<len(layer1.weights[r]) {
			layer1.weights[r][c] = rand.float64_range(0.,5.)
		}
	}

	output_layer: Layer
	output_layer.weights = create_matrix(len(Classification), 5)
	defer {
		clean_matrix(output_layer.weights)
	}
	for r in 0..<len(output_layer.weights ) {
		for c in 0..<len(output_layer.weights[r]) {
			output_layer.weights[r][c] = rand.float64_range(0.,5.)
		}
	}

	layers: [NUM_LAYERS]Layer
	layers[0] = layer1
	layers[1] = output_layer

	//Process input

	input_matrix := make_slice(Matrix, ITERATIONS)
	for &row in input_matrix {
		row = make_slice([]f64, 4)
	}
	defer {
		clean_matrix(input_matrix)
	}
	index := 0
	for i in 0..<ITERATIONS {
		for j in 0..<4 {
			val := rand.float64_range(0.,100.)
			input_matrix[i][j] = val
		}
		index += 1
	}

	expected_vector := make_slice([]int, ITERATIONS)
	index = 0
	for &entry, i in expected_vector {
		index, max := find_min(input_matrix[i])
		entry = index
	}
	
	//Calculate layers

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	for &input, index in input_matrix {
		input = normalize_vector(input)
		chosen_output, o, layer1_neurons := foward_prop(input, layers)
		if chosen_output > -1 {
			expected_o := expected_vector[index]
			loss, accuracy := back_prop(expected_o, o[:], chosen_output, layers, layer1_neurons)
			if index == ITERATIONS -1  { 
				fmt.println("\t======ITERATION: ", index)
				fmt.println("STATS!!!")
				fmt.println("CHOSEN: ",cast(Classification) chosen_output)
				fmt.println("EXPTED: ",Classification(expected_vector[index]))
				fmt.println("LOSS: ", loss)
				fmt.println("ACCURACY: ", accuracy * 100)
			}
		}
	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	/* 
	running := true
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

		o, a, b := foward_prop(data[:], layers)
		fmt.println(Classification(o))
		//TODO: Add some runtime training this means we need to figure out the expted value for a given state
	} */
}

foward_prop :: proc(input: []f64, layers: [NUM_LAYERS]Layer) -> (int, [4]f64, []f64) {
	layer1_neurons := dot(input, layers[0].weights)
	defer delete(layer1_neurons)
	for &n in layer1_neurons {
		/* n += bias */
		n = relu(n)
	}
	layer1_neurons = normalize_vector(layer1_neurons[:])

	output_layer := dot(layer1_neurons, layers[1].weights)
	defer delete(output_layer)
	for &n in output_layer {
		/* n += bias */
	}
	output_layer = normalize_vector(output_layer[:])
	o := soft_max(output_layer[:])

	//Make decision
	sum:f64 = 0.
	chosen_output := -1
	curr_max := -1.
	for v, i in o {
		if v > curr_max {
			curr_max = v
			chosen_output = i
		}
		sum += v
	}

	return chosen_output, o, layer1_neurons
}

back_prop :: proc(expected_o: int, o: []f64, chosen_output: int, layers: [NUM_LAYERS]Layer, layer1_neurons: []f64) -> (loss: f64, accuracy: f64){
	//Calc loss

	loss = cross_entropy_loss(o[expected_o])
	accuracy = 1 - math.abs(cross_entropy_loss(o[chosen_output]))

	//Back Prop

	alpha := 0.1
	//UPDATE OUTPUT LAYER////////
	for &row, r in layers[1].weights[chosen_output] {
		new_weight := row - alpha * ( loss)
		row = new_weight
	}

	//UPDATE LAYER 1 //////////
	layer1_neuron_loss := make_slice([]f64, len(layer1_neurons))
	defer delete(layer1_neuron_loss)
	weights_to_update_t := transpose(layers[0].weights) 
	defer {
		clean_matrix(weights_to_update_t)
	}
	for row, r in weights_to_update_t {
		for col, c in weights_to_update_t[r] {
			weight := weights_to_update_t[r][c]
			d_a := d_relu(layer1_neurons[c])
			neuron_loss := weight * loss * d_a
			layer1_neuron_loss[c] = neuron_loss
		}
	}

	for row, r in weights_to_update_t {
		for &col, c in weights_to_update_t[r] {
			new_weight := col - alpha * (layer1_neuron_loss[r] * loss)
			col = new_weight
		}
	}
	new_weights_to_update := transpose(weights_to_update_t)
	defer {
		clean_matrix(new_weights_to_update)
	}
	return
}

weighted_sum :: proc(input, weights: []f64, bias: f64) -> f64 {
	assert(len(input) == len(weights))
	sum:f64 = 0.
	for i in 0..<len(input) {
		sum += input[i] * weights[i]
	}
	return sum + bias
}

relu :: proc(z: f64) -> f64 {
	return max(0., z)
}

d_relu :: proc(z: f64) -> f64 {
	return z >= 0. ? 1. : 0.
}

soft_max :: proc(x: []f64) -> [len(Classification)]f64{
	assert(len(x) == len(Classification))
	//For a given vector calculate:
	//sum of math.exp of each value
	//Then for each value it is exp(i)/ sum of exp(i)
	ret:[len(Classification)]f64
	exp_sum:f64 = 0.
	for val in x {
		exp_sum += math.exp(val)
	}
	for val, index in x {
		ret[index] = math.exp(val)/exp_sum
	}
	return ret
}

calc_loss :: proc() {
	//
}

//Normally uses one-hot encoding to get the loss
//sum from 1 - number of classifications of y*log(y-hat) where y is the one-hot encoded entry and y-hat is the chosen class from prediction
cross_entropy_loss :: proc(predicted_val: f64) -> f64 {
	return -(math.log10(predicted_val))
}

dot :: proc(v: []f64, m:Matrix) -> []f64 {
	assert(len(v) == len(m[0]))
	output := make_slice([]f64, len(m))
	
	for r in 0..<len(m) {
		curr_sum := 0.
		for c in 0..<len(m[r]) {
			curr_sum += v[c] * m[r][c]
		}
		output[r] = curr_sum
	}

	return output
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

create_matrix :: proc(rows, cols: int) -> Matrix {
	m := make_slice(Matrix , rows)
	for &row in m {
		row = make_slice([]f64, cols)
	}
	return m 
}

transpose :: proc(input_matrix: Matrix) -> Matrix {
	input_matrix_t := make_slice(Matrix, len(input_matrix[0]))
	for &row in input_matrix_t {
		row = make_slice([]f64, len(input_matrix))
	}

	for c in 0..<len(input_matrix[0]) {
		for r in 0..<len(input_matrix) {
			input_matrix_t[c][r] = input_matrix[r][c]
		}
	}
	return input_matrix_t
}

find_min :: proc(vector: []f64) -> (int, f64) {
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

temp :: proc() {
	x := 500.
	z := math.exp_f64(x)
	fmt.println(z)
}

print_matrix :: proc(m: Matrix) {
	for row in m{
		fmt.printfln("%v", row)
	}
}

clean_matrix :: proc(m: Matrix) {
		for &row in m {
			delete(row)
		}
		delete(m)
}
