package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:strconv"
import "core:mem"
import "core:math"
import "core:math/linalg"
import "core:math/rand"

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

INTERATIONS :: 1000000

main :: proc() {

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
	layer1_weights := create_matrix(5,4)
	defer {
		clean_matrix(layer1_weights)
	}
	for r in 0..<len(layer1_weights) {
		for c in 0..<len(layer1_weights[r]) {
			layer1_weights[r][c] = rand.float64_range(0.,5.)
		}
	}

	output_layer_weights := create_matrix(len(Classification), 5)
	defer {
		clean_matrix(output_layer_weights)
	}
	for r in 0..<len(output_layer_weights ) {
		for c in 0..<len(output_layer_weights[r]) {
			output_layer_weights[r][c] = rand.float64_range(0.,5.)
		}
	}

	//Process input

	input_matrix := make_slice([][]f64, INTERATIONS)
	for &row in input_matrix {
		row = make_slice([]f64, 4)
	}
	defer {
		clean_matrix(input_matrix)
	}
	index := 0
	for i in 0..<INTERATIONS {
		for j in 0..<4 {
			val := rand.float64_range(0.,100.)
			input_matrix[i][j] = val
		}
		index += 1
	}

	expected_vector := make_slice([]int, INTERATIONS)
	index = 0
	for &entry, i in expected_vector {
		index, max := find_max(input_matrix[i])
		entry = index
	}
	
	//Calculate layers

	for &input, index in input_matrix {
		input = normalize_vector(input)
		chosen_output, o, layer1_neurons := foward_prop(input, layer1_weights, output_layer_weights)
		if chosen_output > -1 {

			//Calc loss
			expected_o := expected_vector[index]

			loss := cross_entropy_loss(o[expected_o])
			accuracy := 1 - math.abs(cross_entropy_loss(o[chosen_output]))

			//Back Prop

			alpha := 0.1
			//UPDATE OUTPUT LAYER////////
			for &row, r in output_layer_weights[chosen_output] {
				new_weight := row - alpha * ( loss)
				row = new_weight
			}

			//UPDATE LAYER 1 //////////
			layer1_neuron_loss := make_slice([]f64, len(layer1_neurons))
			defer delete(layer1_neuron_loss)
			weights_to_update_t := transpose(layer1_weights) 
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
			if index == INTERATIONS -1  { 
				fmt.println("\t======ITERATION: ", index)
				fmt.println("STATS!!!")
				fmt.println("CHOSEN: ",cast(Classification) chosen_output)
				fmt.println("EXPTED: ",Classification(expected_vector[index]))
				fmt.println("LOSS: ", loss)
				fmt.println("ACCURACY: ", accuracy * 100)
			}
		}
	}
	
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

		o, a, b := foward_prop(data[:], layer1_weights, output_layer_weights)
		fmt.println(Classification(o))
	}
}

foward_prop :: proc(input: []f64, layer1_weights, output_layer_weights: [][]f64) -> (int, [4]f64, []f64) {
	layer1_neurons := dot(input, layer1_weights)
	defer delete(layer1_neurons)
	for &n in layer1_neurons {
		/* n += bias */
		n = relu(n)
	}
	layer1_neurons = normalize_vector(layer1_neurons[:])

	output_layer := dot(layer1_neurons, output_layer_weights)
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

back_prop :: proc(weights_to_update: [][]f64, neurons: []f64, loss: f64) {
	
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

dot :: proc(v: []f64, m:[][]f64) -> []f64 {
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

create_matrix :: proc(rows, cols: int) -> [][]f64 {
	m := make_slice([][]f64 , rows)
	for &row in m {
		row = make_slice([]f64, cols)
	}
	return m 
}

transpose :: proc(input_matrix: [][]f64) -> [][]f64 {
	input_matrix_t := make_slice([][]f64, len(input_matrix[0]))
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

find_max :: proc(vector: []f64) -> (int, f64) {
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

temp :: proc() {
	x := 500.
	z := math.exp_f64(x)
	fmt.println(z)
}

print_matrix :: proc(m: [][]f64) {
	for row in m{
		fmt.printfln("%v", row)
	}
}

clean_matrix :: proc(m: [][]f64) {
		for &row in m {
			delete(row)
		}
		delete(m)
}
