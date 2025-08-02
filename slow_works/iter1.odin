package slow_works

import "../constants"

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

classification_counts := [4]int{}
expected_counts := [4]int{}
accuracy_count := 0

//MODEL PARAMS

generate_input :: proc() -> (Matrix, []int) {
	//Process input
	input_matrix := make_slice(Matrix, constants.ITERATIONS)
	for &row in input_matrix {
		row = make_slice([]f64, 4)
	}
	//TODO: Memory leak
	expected_vector := make_slice([]int, constants.ITERATIONS)
	index := 0
	for i in 0..<constants.ITERATIONS {
		for j in 0..<4 {
			val := rand.float64_range(0.,100.)
			input_matrix[i][j] = val
		}

		index, max := find_min(input_matrix[i])
		expected_vector[i] = index
		index += 1
	}

	return input_matrix, expected_vector
}

init_network :: proc() -> [constants.NUM_LAYERS + 1]Layer {
	//Init layers
	layer1: Layer
	layer1.weights = create_matrix(constants.H_NUM_NEURONS,4)
	for r in 0..<len(layer1.weights) {
		for c in 0..<len(layer1.weights[r]) {
			layer1.weights[r][c] = rand.float64_range(-1.,1.)
		}
	}
	fmt.println("LAYER1: ", layer1, "\n")
	/* fmt.println(layer1.weights) */

	output_layer: Layer
	output_layer.weights = create_matrix(len(Classification), constants.H_NUM_NEURONS)
	for r in 0..<len(output_layer.weights ) {
		for c in 0..<len(output_layer.weights[r]) {
			output_layer.weights[r][c] = rand.float64_range(-1.,1.)
		}
	}

	fmt.println("OUTPUT_LAYER: ", output_layer, "\n")
	layers: [constants.NUM_LAYERS + 1]Layer
	layers[0] = layer1
	layers[1] = output_layer
	return layers
}

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


	/* fmt.println(input_matrix) */

	input_matrix, expected_vector := generate_input() 
	defer {
		clean_matrix(input_matrix)
	}

	layers := init_network()
	defer {
		for &layer in layers {
			clean_matrix(layer.weights)
		}
	}
	
	//Calculate layers

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	for &input, index in input_matrix {
		input = normalize_vector(input)
		chosen_output, o, layer1_neurons := foward_prop(input, layers)
		/* fmt.println("O: ", o) */
		if chosen_output > -1 {
			expected_o := expected_vector[index]
			loss, new_output_layer, new_hidden_layer := back_prop(expected_o, o[:], chosen_output, layers, layer1_neurons)
			fmt.println(o)
			classification_counts[chosen_output] += 1
			expected_counts[expected_o] += 1
			if expected_o == chosen_output {
				accuracy_count += 1
			}
			layers[1] = new_output_layer
			layers[0] = new_hidden_layer
			if index == constants.ITERATIONS -1  { 
				/* fmt.println("\t======ITERATION: ", index)
				fmt.println("STATS!!!")
				fmt.println("CHOSEN: ",cast(Classification) chosen_output)
				fmt.println("EXPTED: ",Classification(expected_vector[index]))
				fmt.println("LOSS: ", loss)
				fmt.println("ACCURACY: ", accuracy * 100) */
			}
		}
	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	fmt.printfln("CLASSIFICATION COUNTS: %v", classification_counts)
	fmt.printfln("EXPECTED COUNTS: %v", expected_counts)
	accuracy := (f32(accuracy_count)/f32(constants.ITERATIONS)) * 100
	fmt.println("ACCURACY: ",accuracy)


	
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

		o, a, b := foward_prop(data[:], layers)
		fmt.println(Classification(o))
		//TODO: Add some runtime training this means we need to figure out the expted value for a given state
	} */
}

forward_hidden_layer :: proc(input: []f64, hidden_layer: Layer) -> []f64  {
	layer1_neurons := dot(input, hidden_layer.weights)
	/* defer delete(layer1_neurons) */
	for &n in layer1_neurons {
		/* n += bias */
		n = relu(n)
	}
	/* layer1_neurons = normalize_vector(layer1_neurons[:]) */
	return layer1_neurons
}

forward_output :: proc(prev_layer_neurons: []f64, output_layer: Layer) -> [len(Classification)]f64 {
	neurons := dot(prev_layer_neurons, output_layer.weights)
	/* defer delete(output_layer) */
	for &n in neurons {
		/* n += bias */
	}
	neurons = normalize_vector(neurons)
	/* output_layer = normalize_vector(output_layer[:]) */
	o := soft_max(neurons[:])
	return o
}

foward_prop :: proc(input: []f64, layers: [constants.NUM_LAYERS + 1]Layer) -> (int, [4]f64, []f64) {

	layer1_neurons := forward_hidden_layer(input, layers[0])
	defer delete(layer1_neurons)
	o := forward_output(layer1_neurons, layers[1])

	//Make decision
	sum:f64 = 0.
	chosen_output := -1
	curr_max := -99999.
	for v, i in o {
		if v > curr_max {
			curr_max = v
			chosen_output = i
		}
		sum += v
	}

	return chosen_output, o, layer1_neurons
}

output_layer_train :: proc(loss: f64, output_neurons: []f64, chosen_output:int, output_layer: Layer) -> Layer {
	_output_layer := output_layer
	one_hot_arr:[len(Classification)]f64
	one_hot_arr[chosen_output] = 1.
	total_loss := 0.
	for i in 0..<len(Classification) {
		mse := math.pow(output_neurons[i] - one_hot_arr[i], 2)
		total_loss += mse
	}
	for &row, r in _output_layer.weights {
		for &col, c in _output_layer.weights[r] {
			new_weight := col - constants.ALPHA * (loss)
			/* new_weight := col - constants.ALPHA * (output_neurons[c] - one_hot_arr[c]) */
			col = new_weight
		}
	}

	return _output_layer
}

hidden_layer_train :: proc(loss:f64, layer_neurons: []f64, hidden_layer: Layer) -> Matrix {
	layer1_neuron_loss := make_slice([]f64, len(layer_neurons))
	defer delete(layer1_neuron_loss)
	weights_to_update_t := transpose(hidden_layer.weights) 
	defer {
		clean_matrix(weights_to_update_t)
	}
	for row, r in weights_to_update_t {
		for col, c in weights_to_update_t[r] {
			weight := weights_to_update_t[r][c]
			//TODO: I believe this is the incorrect value to d_relu since this value will ALSO be 1 since our layer_neuron which is activated by relu is always >= 0
			d_activation := d_relu(layer_neurons[c])
			neuron_loss := weight * loss * d_activation
			layer1_neuron_loss[c] = neuron_loss
		}
	}

	for row, r in weights_to_update_t {
		for &col, c in weights_to_update_t[r] {
			new_weight := col - constants.ALPHA * (layer1_neuron_loss[r] * loss)
			col = new_weight
		}
	}
	new_weights_to_update := transpose(weights_to_update_t)

	return new_weights_to_update
}

back_prop :: proc(expected_o: int, output_neurons: []f64, chosen_output: int, layers: [constants.NUM_LAYERS + 1]Layer, layer1_neurons: []f64) -> (loss: f64, new_output_layer: Layer, new_hidden_layer: Layer) {
	//Calc loss

	loss = cross_entropy_loss(output_neurons[chosen_output])

	//Back Prop

	//UPDATE OUTPUT LAYER////////
	new_output_layer = output_layer_train(loss, output_neurons, chosen_output, layers[1])
	/* for &row, r in layers[0].weights[chosen_output] {
		new_weight := row - constants.ALPHA * (loss)
		row = new_weight
	}
	new_output_layer = layers[0] */


	//UPDATE LAYER 1 //////////
	new_hidden_layer = layers[0]
	new_hidden_layer.weights = hidden_layer_train(loss, layer1_neurons, layers[0])
	/* layer1_neuron_loss := make_slice([]f64, len(layer1_neurons))
	defer delete(layer1_neuron_loss)
	weights_to_update_t := transpose(layers[1].weights) 
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
			new_weight := col - constants.ALPHA * (layer1_neuron_loss[r] * loss)
			col = new_weight
		}
	}
	new_weights_to_update := transpose(weights_to_update_t)
	new_hidden_layer.weights = new_weights_to_update
	defer {
		clean_matrix(new_weights_to_update)
	} */

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

sigmoid :: proc(x: f64) -> f64 {
	return 1/(1 + math.exp(-x))
}

d_sigmoid :: proc(x: f64) -> f64 {
	return sigmoid(x) * (1 - sigmoid(x))
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
	//TODO: Leaking memory here
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
		val = (numerator/denom) * range - 1.
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

find_min :: proc(vector: []$T) -> (int, T) {
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
