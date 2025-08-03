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
}

State :: struct {
	wealth: f64,
	health: f64,
	supplies: f64,
	strength: f64,
}

Layer :: struct {
	weights: Matrix,
	weighted_sums: []f64,
	activated_neurons: []f64,
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
	layer1: Layer
	layer1.weights = create_matrix(constants.H_NUM_NEURONS,4)
	layer1.weighted_sums = make_slice([]f64, constants.H_NUM_NEURONS)
	layer1.activated_neurons = make_slice([]f64, constants.H_NUM_NEURONS)
	for r in 0..<len(layer1.weights) {
		for c in 0..<len(layer1.weights[r]) {
			layer1.weights[r][c] = rand.float64_range(-1.,1.)
		}
	}
	fmt.println("LAYER1: ", layer1, "\n")

	output_layer: Layer
	output_layer.weights = create_matrix(len(Classification), constants.H_NUM_NEURONS)
	output_layer.weighted_sums = make_slice([]f64, constants.O_NUM_NEURONS)
	output_layer.activated_neurons = make_slice([]f64, constants.O_NUM_NEURONS)
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

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	for &input, index in input_matrix {
		input = normalize_vector(input)
		chosen_output, o:= foward_prop(input, &layers)
		/* fmt.println(o) */
		if chosen_output > -1 {
			expected_o := expected_vector[index]
			loss, new_output_layer := back_prop(expected_o, chosen_output, o[:], &layers)
			classification_counts[chosen_output] += 1
			expected_counts[expected_o] += 1
			if expected_o == chosen_output {
				accuracy_count += 1
			}
			/* layers[1] = new_output_layer
			layers[0] = new_hidden_layer */
		}
	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	fmt.printfln("CLASSIFICATION COUNTS: %v", classification_counts)
	fmt.printfln("EXPECTED COUNTS: %v", expected_counts)
	accuracy := (f32(accuracy_count)/f32(constants.ITERATIONS)) * 100
	fmt.println("ACCURACY: ",accuracy)


}

forward_hidden_layer :: proc(input: []f64, hidden_layer: ^Layer) {
	dot(input, hidden_layer)
	hidden_layer.weighted_sums = normalize_vector(hidden_layer.weighted_sums[:])
	for n, i in hidden_layer.weighted_sums {
		hidden_layer.activated_neurons[i] = relu(n)
	}
}

forward_output :: proc(prev_layer_neurons: []f64, output_layer: ^Layer) -> [len(Classification)]f64 {
	dot(prev_layer_neurons, output_layer)
	neurons := normalize_vector(output_layer.weighted_sums[:])
	o := soft_max(neurons)
	return o
}

foward_prop :: proc(input: []f64, layers: ^[constants.NUM_LAYERS + 1]Layer) -> (int, [4]f64) {
	forward_hidden_layer(input, &layers[0])
	o := forward_output(layers[0].activated_neurons[:], &layers[1])

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

	return chosen_output, o
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
			/* new_weight := col - constants.ALPHA * (loss) */
			new_weight := col - constants.ALPHA * (output_neurons[c] - one_hot_arr[c])
			col = new_weight
		}
	}
	return _output_layer
}

//TODO: I think this is our biggest issue right now look at this again and figure out a better way to do it
hidden_layer_train :: proc(loss:f64, hidden_layer: ^Layer) {

	for row, r in hidden_layer.weights {
		for &col, c in hidden_layer.weights[r] {
			d_activation := d_relu(hidden_layer.weighted_sums[r])
			neuron_loss := col * loss * d_activation
			col = col - constants.ALPHA * (neuron_loss)
		}
	}
}

back_prop :: proc(expected_o, chosen_output: int, output_neurons: []f64, layers: ^[constants.NUM_LAYERS + 1]Layer) -> (loss: f64, new_output_layer: Layer) {
	loss = cross_entropy_loss(output_neurons[chosen_output])
	output_layer_train(loss, output_neurons, chosen_output, layers[1])
	hidden_layer_train(loss, &layers[0])
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

cross_entropy_loss :: proc(predicted_val: f64) -> f64 {
	return -(math.log10(predicted_val))
}

dot :: proc(v: []f64, layer: ^Layer) {
	assert(len(v) == len(layer.weights[0]))
	//TODO: Leaking memory here
	
	for r in 0..<len(layer.weights) {
		curr_sum := 0.
		for c in 0..<len(layer.weights[r]) {
			curr_sum += v[c] * layer.weights[r][c]
		}
		layer.weighted_sums[r] = curr_sum
	}
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
	if denom > 0 {
		for &val in vector {
			numerator := val - min
			range := 1. - -1.
			val = (numerator/denom) * range - 1.
		}
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
